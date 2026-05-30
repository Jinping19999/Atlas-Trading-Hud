"""
Atlas Trading HUD — FastAPI Backend
Wraps scanner_polygon.py to serve signals over HTTP.

Endpoints:
  POST /scan       — trigger a fresh scan, return results
  GET  /signals    — return the most recent cached scan
  GET  /health     — liveness check

Run locally:
  1. Make sure .env file has POLYGON_API_KEY=your_key
  2. pip install -r requirements.txt
  3. uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.
"""

import os
import json
import time
import asyncio
import datetime
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Load .env file if present (keeps API keys out of source code)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    # python-dotenv not installed — fall back to manual env vars
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Import V5 scanner (yfinance-based, no API key needed) ─
import scanner_v5

# V1 scanner (Polygon-based) — optional, only if scanner_polygon.py exists
try:
    import scanner_polygon as scanner
    scanner.POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
    scanner.POLYGON_TIER = os.environ.get("POLYGON_TIER", "free")
    scanner.CALL_DELAY = 13.0 if scanner.POLYGON_TIER == "free" else 0.1
    scanner._limiter = scanner.RateLimiter(scanner.CALL_DELAY)
    _HAS_V1_SCANNER = True
except (ImportError, Exception):
    _HAS_V1_SCANNER = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("atlas-api")

# ── In-memory state (replaced by SQLite in step 5) ──────────────────────────
_last_scan: dict = {
    "timestamp": None,       # ISO string of when the scan ran
    "date": None,            # trading date the scan covers
    "signals": [],           # list of signal dicts
    "primary": [],           # tickers with primary signal today
    "secondary": [],         # tickers with secondary signal today
    "watchlist": [],         # tickers with score >= 7
    "meta": {},              # scan metadata (duration, ticker count, etc.)
}
_scan_lock = asyncio.Lock()  # prevent concurrent scans
_scan_running = False

# V5 scan state
_last_v5_scan: dict = {
    "timestamp": None,
    "date": None,
    "regime": None,
    "sectors": [],
    "signals": [],
    "weeklyPerformers": [],
    "personalSignals": [],
    "personalStatus": [],
    "meta": {},
}
_v5_scan_running = False

# ── Cached scan persistence (JSON on disk) ───────────────────────────────────
SCAN_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".scan_cache")
os.makedirs(SCAN_CACHE_DIR, exist_ok=True)


def _scan_cache_path() -> str:
    today = datetime.date.today().isoformat()
    return os.path.join(SCAN_CACHE_DIR, f"scan_{today}.json")


def _load_cached_scan() -> Optional[dict]:
    """Load today's cached scan from disk if it exists."""
    path = _scan_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load scan cache: {e}")
    return None


def _save_scan_cache(scan_result: dict):
    """Persist scan result to disk so GET /signals survives server restarts."""
    path = _scan_cache_path()
    try:
        with open(path, "w") as f:
            json.dump(scan_result, f, default=str)
        log.info(f"Scan cached to {path}")
    except Exception as e:
        log.warning(f"Failed to save scan cache: {e}")


def _v5_cache_path() -> str:
    today = datetime.date.today().isoformat()
    return os.path.join(SCAN_CACHE_DIR, f"v5_scan_{today}.json")


def _load_cached_v5_scan() -> Optional[dict]:
    path = _v5_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load V5 scan cache: {e}")
    return None


def _save_v5_scan_cache(scan_result: dict):
    path = _v5_cache_path()
    try:
        with open(path, "w") as f:
            json.dump(scan_result, f, default=str)
        log.info(f"V5 scan cached to {path}")
    except Exception as e:
        log.warning(f"Failed to save V5 scan cache: {e}")


# ── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, load today's cached scans if available."""
    global _last_scan, _last_v5_scan
    if _HAS_V1_SCANNER:
        cached = _load_cached_scan()
        if cached:
            _last_scan = cached
            log.info(f"Loaded cached V1 scan from {cached.get('timestamp', '?')} "
                     f"with {len(cached.get('signals', []))} signals")
    cached_v5 = _load_cached_v5_scan()
    if cached_v5:
        _last_v5_scan = cached_v5
        log.info(f"Loaded cached V5 scan from {cached_v5.get('timestamp', '?')} "
                 f"with {len(cached_v5.get('signals', []))} signals")
    yield


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Atlas Trading HUD API",
    version="0.1.0",
    description="Backend for the Atlas systematic trading dashboard",
    lifespan=lifespan,
)

# CORS — allow the frontend from any origin (single-user tool)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the HUD frontend from the same origin ─────────────────────────────
_HUD_PATH = Path(__file__).parent / "v5_trading_hud.html"

@app.get("/", include_in_schema=False)
async def serve_hud():
    """Serve the trading HUD at the root URL."""
    if _HUD_PATH.exists():
        return FileResponse(_HUD_PATH, media_type="text/html")
    raise HTTPException(404, "v5_trading_hud.html not found")


# ── Pydantic models ─────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    """Optional overrides for a scan."""
    tickers: Optional[list[str]] = None   # override default ticker list
    force: bool = False                   # re-scan even if today's cache exists


class SignalOut(BaseModel):
    ticker: str
    close: float
    vah: Optional[float]
    ema21: float
    rsi: Optional[float]
    above_vah: int
    vs_vah_pct: float
    vs_ema21_pct: float
    atr_ratio: float
    f_sec: int
    f_mkt: int
    f_rs: int
    today_primary: int
    today_secondary: int
    days_since_primary: Optional[int]
    days_since_secondary: Optional[int]
    recent_primary_count: int
    recent_secondary_count: int
    score: int
    grade: str
    sig_type: str


class ScanResponse(BaseModel):
    timestamp: Optional[str]
    date: Optional[str]
    signals: list[dict]
    primary: list[str]
    secondary: list[str]
    watchlist: list[str]
    meta: dict


class HealthResponse(BaseModel):
    status: str
    polygon_key_set: bool
    last_scan: Optional[str]
    scan_running: bool


# ── Core V1 scan logic (runs in thread pool — scanner is blocking I/O) ─────
def _run_scan_sync(tickers: Optional[list[str]] = None) -> dict:
    """
    Execute scanner_polygon.py logic synchronously.
    Returns the structured scan result dict.
    """
    if not _HAS_V1_SCANNER:
        raise RuntimeError("V1 scanner (scanner_polygon) not available. Use /v5/scan instead.")

    t0 = time.time()

    # Validate Polygon key
    if not scanner.POLYGON_API_KEY:
        raise RuntimeError(
            "POLYGON_API_KEY not set. "
            "Set it as an environment variable before starting the server."
        )

    # Use provided tickers or scanner defaults
    ticker_list = tickers or scanner.TICKERS

    # Recalculate date range (scanner uses module-level globals)
    start = (datetime.date.today() - datetime.timedelta(days=550)).strftime("%Y-%m-%d")
    end = datetime.date.today().strftime("%Y-%m-%d")

    log.info(f"Starting scan: {len(ticker_list)} tickers, {start} → {end}")

    # Fetch data
    data, idx = scanner.fetch_all(ticker_list, start, end)
    log.info(f"Data fetched: {len(idx)} trading days")

    # Build signals
    results = []
    skipped = []
    for t in ticker_list:
        try:
            df = scanner.build_signals(t, data)
            if df is None or len(df.dropna(subset=["close"])) < 80:
                skipped.append(t)
                continue
            s = scanner.current_state(t, df)
            if s is None:
                skipped.append(t)
                continue
            score, grade, sig_type = scanner.grade_setup(s)
            s["score"] = score
            s["grade"] = grade
            s["sig_type"] = sig_type
            results.append(s)
        except Exception as e:
            log.warning(f"  {t}: ERROR — {e}")
            skipped.append(t)

    results.sort(key=lambda x: x["score"], reverse=True)

    duration = round(time.time() - t0, 1)

    primary_today = [s["ticker"] for s in results if s["today_primary"]]
    secondary_today = [s["ticker"] for s in results if s["today_secondary"]]
    watch_now = [
        s["ticker"] for s in results
        if s["score"] >= 7 and not s["today_primary"] and not s["today_secondary"]
    ]

    scan_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "date": end,
        "signals": results,
        "primary": primary_today,
        "secondary": secondary_today,
        "watchlist": watch_now,
        "meta": {
            "tickers_scanned": len(ticker_list),
            "signals_found": len(results),
            "skipped": skipped,
            "duration_seconds": duration,
            "polygon_tier": scanner.POLYGON_TIER,
            "data_range": f"{idx[0].date()} → {idx[-1].date()}" if len(idx) else "N/A",
        },
    }

    log.info(
        f"Scan complete: {len(results)} signals, "
        f"{len(primary_today)} primary, {len(secondary_today)} secondary "
        f"({duration}s)"
    )

    return scan_result


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Server health and status check."""
    return {
        "status": "ok",
        "polygon_key_set": bool(scanner.POLYGON_API_KEY),
        "last_scan": _last_scan.get("timestamp"),
        "scan_running": _scan_running,
    }


@app.post("/scan", response_model=ScanResponse)
async def scan(req: ScanRequest = ScanRequest()):
    """
    Trigger a fresh scan. Returns signals sorted by score (descending).

    - If today's scan already exists and `force` is false, returns cached result.
    - Scan runs in a background thread to avoid blocking the event loop.
    - Only one scan can run at a time (mutex).
    """
    global _last_scan, _scan_running

    # Return cached if today's scan exists and not forced
    if not req.force and _last_scan.get("date") == datetime.date.today().isoformat():
        log.info("Returning cached scan (same day, force=false)")
        return _last_scan

    # Acquire lock — only one scan at a time
    if _scan_running:
        raise HTTPException(status_code=409, detail="Scan already in progress")

    async with _scan_lock:
        _scan_running = True
        try:
            # Run blocking scanner in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _run_scan_sync, req.tickers
            )
            _last_scan = result
            _save_scan_cache(result)
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            log.error(f"Scan failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Scan failed: {e}")
        finally:
            _scan_running = False


@app.get("/signals", response_model=ScanResponse)
async def get_signals():
    """
    Return the most recent scan results.
    Does NOT trigger a new scan — use POST /scan for that.
    Returns 404 if no scan has been run today.
    """
    if not _last_scan.get("timestamp"):
        raise HTTPException(
            status_code=404,
            detail="No scan results available. Run POST /scan first."
        )
    return _last_scan


@app.get("/signals/{ticker}")
async def get_signal_by_ticker(ticker: str):
    """Return signal data for a specific ticker from the last scan."""
    if not _last_scan.get("signals"):
        raise HTTPException(status_code=404, detail="No scan results available.")

    ticker_upper = ticker.upper()
    for s in _last_scan["signals"]:
        if s["ticker"] == ticker_upper:
            return s

    raise HTTPException(
        status_code=404,
        detail=f"Ticker {ticker_upper} not found in last scan results."
    )


# ── V5 Endpoints ────────────────────────────────────────────────────────────

class V5ScanRequest(BaseModel):
    """Configuration for a V5 scan."""
    top_n: int = 2                              # how many top sectors to include
    personal_watchlist: list[str] = []           # tickers to scan outside sector filter
    force: bool = False                         # re-scan even if today's cache exists


class V5ScanResponse(BaseModel):
    timestamp: Optional[str]
    date: Optional[str]
    regime: Optional[dict]
    sectors: list[dict]
    signals: list[dict]
    weeklyPerformers: list[dict]
    personalSignals: list[dict]
    personalStatus: list[dict]
    meta: dict


def _run_v5_scan_sync(top_n: int, personal_watchlist: list[str]) -> dict:
    """Execute V5 scan synchronously (called from thread pool)."""
    return scanner_v5.v5_scan(
        tickers_by_sector=None,  # use default universe
        top_n=top_n,
        personal_watchlist=personal_watchlist,
    )


@app.post("/v5/scan", response_model=V5ScanResponse)
async def v5_scan(req: V5ScanRequest = V5ScanRequest()):
    """
    Run V5 Adaptive Momentum scan.

    - Returns cached result if today's scan exists and force=false.
    - Only one scan can run at a time (shares mutex with V1 scan).
    """
    global _last_v5_scan, _v5_scan_running

    # Return cached if today's V5 scan exists and not forced
    if not req.force and _last_v5_scan.get("date") == datetime.date.today().isoformat():
        log.info("Returning cached V5 scan (same day, force=false)")
        return _last_v5_scan

    if _scan_running or _v5_scan_running:
        raise HTTPException(status_code=409, detail="A scan is already in progress")

    async with _scan_lock:
        _v5_scan_running = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _run_v5_scan_sync, req.top_n, req.personal_watchlist
            )
            _last_v5_scan = result
            _save_v5_scan_cache(result)
            return result
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            log.error(f"V5 Scan failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"V5 scan failed: {e}")
        finally:
            _v5_scan_running = False


@app.get("/v5/signals", response_model=V5ScanResponse)
async def get_v5_signals():
    """Return the most recent V5 scan results (does NOT trigger a new scan)."""
    if not _last_v5_scan.get("timestamp"):
        raise HTTPException(
            status_code=404,
            detail="No V5 scan results available. Run POST /v5/scan first."
        )
    return _last_v5_scan


@app.get("/v5/signals/{ticker}")
async def get_v5_signal_by_ticker(ticker: str):
    """Return V5 signal data for a specific ticker from the last scan."""
    if not _last_v5_scan.get("signals"):
        raise HTTPException(status_code=404, detail="No V5 scan results available.")

    ticker_upper = ticker.upper()
    for s in _last_v5_scan["signals"]:
        if s["tk"] == ticker_upper:
            return s

    # Also check personal signals
    for s in _last_v5_scan.get("personalSignals", []):
        if s["tk"] == ticker_upper:
            return s

    raise HTTPException(
        status_code=404,
        detail=f"Ticker {ticker_upper} not found in V5 scan results."
    )


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
