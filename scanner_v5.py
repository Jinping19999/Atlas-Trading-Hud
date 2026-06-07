"""
V5 Adaptive Momentum Breakout Scanner — Python Backend (yfinance edition)
Uses yfinance bulk downloads — entire scan completes in ~30s vs 55min on Polygon free tier.

Strategy rules:
  1. Regime: SPY > EMA50 (bull market filter)
  2. Sector rotation: Top-N sectors by 63-day RS vs SPY, must be in uptrend
  3. Price > EMA21
  4. RSI 40–70 (momentum sweet spot)
  5. RS₂₁ > 8% vs SPY (relative strength)
  6. Conviction: pullback in last 3 bars OR rvol ≥ 1.25×

Returns signals in the exact JSON format the Atlas HUD frontend expects.
"""

import os
import json
import time
import datetime
import logging
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("atlas-v5")

# Cache directory for S&P 500 constituent list
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.scan_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)

# ── V5 Strategy Parameters ──────────────────────────────────────────────────
RS_THRESHOLD   = 0.08    # RS₂₁ vs SPY must exceed 8%
RSI_LOW        = 40
RSI_HIGH       = 70
ATR_STOP_MULT  = 2.5
TRAIL_ATR_MULT = 0.5
VOL_CONVICTION = 1.25    # rvol threshold for conviction sizing
SECTOR_LOOKBACK = 63     # ~3 months for sector ranking
MARKET_CLOSE_HOUR = 16   # 4:00 PM ET — after this, today's bar is "confirmed"
MARKET_CLOSE_MINUTE = 5  # 5 min buffer after close for final prints

SECTOR_ETFS = ['XLK', 'XLC', 'XLY', 'XLI', 'XLV', 'XLF', 'XLE', 'XLU', 'XLP', 'XLB', 'XLRE']

SECTOR_NAMES = {
    'XLK': 'Technology', 'XLC': 'Comm Services', 'XLY': 'Cons Discretionary',
    'XLI': 'Industrials', 'XLV': 'Healthcare', 'XLF': 'Financials',
    'XLE': 'Energy', 'XLU': 'Utilities', 'XLP': 'Cons Staples',
    'XLB': 'Materials', 'XLRE': 'Real Estate',
}

# ── GICS Sector → SPDR ETF mapping ──────────────────────────────────────────
_GICS_TO_ETF = {
    'Information Technology': 'XLK',
    'Communication Services': 'XLC',
    'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Consumer Staples': 'XLP',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
}

# Yahoo Finance sector names (slightly different from GICS)
_YF_SECTOR_TO_ETF = {
    'Technology': 'XLK', 'Communication Services': 'XLC',
    'Consumer Cyclical': 'XLY', 'Industrials': 'XLI',
    'Healthcare': 'XLV', 'Financial Services': 'XLF',
    'Energy': 'XLE', 'Utilities': 'XLU',
    'Consumer Defensive': 'XLP', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE',
}


def fetch_sp500_universe() -> dict:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Returns dict mapping sector ETF → [tickers].
    Caches to disk for 7 days to avoid hitting Wikipedia every scan.
    """
    cache_path = os.path.join(_CACHE_DIR, 'sp500_universe.json')

    # Check disk cache (valid for 7 days)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            cache_age = time.time() - cached.get('fetched_at', 0)
            if cache_age < 7 * 86400:  # 7 days
                log.info(f"Using cached S&P 500 list ({len(cached.get('all_tickers', []))} tickers, "
                         f"cached {cache_age / 3600:.0f}h ago)")
                return cached['universe']
        except Exception as e:
            log.warning(f"Failed to load SP500 cache: {e}")

    # Fetch from Wikipedia
    log.info("Fetching S&P 500 constituent list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AtlasHUD/1.0'})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')

        from io import StringIO
        tables = pd.read_html(StringIO(html))
        sp500_df = tables[0]
    except Exception as e:
        log.error(f"Failed to fetch S&P 500 list: {e}")
        # Fall back to cache even if expired
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)['universe']
        raise RuntimeError(f"Cannot fetch S&P 500 list and no cache available: {e}")

    # Build universe dict: sector ETF → [tickers]
    universe = {etf: [] for etf in SECTOR_ETFS}
    all_tickers = []

    for _, row in sp500_df.iterrows():
        symbol = str(row['Symbol']).strip()
        gics_sector = str(row.get('GICS Sector', '')).strip()

        # Fix tickers with dots (BRK.B → BRK-B for yfinance)
        symbol = symbol.replace('.', '-')

        etf = _GICS_TO_ETF.get(gics_sector, None)
        if etf and etf in universe:
            universe[etf].append(symbol)
            all_tickers.append(symbol)

    # Cache to disk
    try:
        with open(cache_path, 'w') as f:
            json.dump({
                'universe': universe,
                'all_tickers': all_tickers,
                'fetched_at': time.time(),
                'count': len(all_tickers),
            }, f)
        log.info(f"Cached S&P 500 list: {len(all_tickers)} tickers across {len(universe)} sectors")
    except Exception as e:
        log.warning(f"Failed to cache SP500 list: {e}")

    return universe


# Build reverse map on demand (lazy, refreshes with universe)
_TICKER_TO_SECTOR = {}

def _ensure_ticker_map(universe: dict):
    """Rebuild ticker→sector map from current universe."""
    global _TICKER_TO_SECTOR
    _TICKER_TO_SECTOR = {}
    for etf, tks in universe.items():
        for tk in tks:
            _TICKER_TO_SECTOR[tk] = etf


# ── Market hours detection ─────────────────────────────────────────────────
def _market_is_closed():
    """
    Check if US market has closed for today (past 4:05 PM ET).
    Returns True if market is closed → today's bar is confirmed.
    Returns False if market is open or pre-market → today's bar is incomplete.
    On weekends, returns True (last Friday's bar is confirmed).
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now_et = datetime.datetime.now(ZoneInfo('America/New_York'))
    # Weekends — last bar is confirmed
    if now_et.weekday() >= 5:
        return True
    # After 4:05 PM ET — today's bar is confirmed
    if now_et.hour > MARKET_CLOSE_HOUR or (
        now_et.hour == MARKET_CLOSE_HOUR and now_et.minute >= MARKET_CLOSE_MINUTE
    ):
        return True
    return False


def _drop_today_if_incomplete(data_dict, keep_intraday=False):
    """
    Handle today's bar depending on market state.
    - Market closed → keep today's bar (it's confirmed).
    - Market open + keep_intraday=True → keep today's partial bar for intraday signals.
    - Market open + keep_intraday=False → drop today's bar (old behaviour).
    Returns (cleaned_dict, bar_date_label, is_intraday) where is_intraday=True
    means the last bar is a live/partial bar during market hours.
    """
    if _market_is_closed():
        # Market closed — today's bar is the confirmed close
        sample = next(iter(data_dict.values()), None)
        bar_date = str(sample.index[-1].date()) if sample is not None and len(sample) > 0 else 'today'
        log.info(f"  Market closed — using confirmed bar: {bar_date}")
        return data_dict, bar_date, False

    if keep_intraday:
        # Market open — KEEP today's partial bar for intraday scanning
        sample = next(iter(data_dict.values()), None)
        bar_date = str(sample.index[-1].date()) if sample is not None and len(sample) > 0 else 'today'
        log.info(f"  Market OPEN — keeping intraday bars (live prices), bar: {bar_date}")
        return data_dict, bar_date, True

    # Market open — drop today's incomplete bar (legacy behaviour)
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    cleaned = {}
    dropped = 0
    for tk, df in data_dict.items():
        if len(df) == 0:
            continue
        last_date = str(df.index[-1].date())
        if last_date == today_str:
            df = df.iloc[:-1]
            dropped += 1
        if len(df) > 0:
            cleaned[tk] = df

    sample = next(iter(cleaned.values()), None)
    bar_date = str(sample.index[-1].date()) if sample is not None and len(sample) > 0 else 'yesterday'
    log.info(f"  Market OPEN — dropped {dropped} partial bars, using last confirmed: {bar_date}")
    return cleaned, bar_date, False


# ── Data fetching via yfinance ──────────────────────────────────────────────
BULK_CHUNK_SIZE = 100   # max tickers per yfinance download call


def fetch_bulk(tickers, start, end):
    """
    Bulk-fetch OHLCV data for multiple tickers via yfinance.
    Returns dict of ticker → DataFrame (columns: Open, High, Low, Close, Volume).
    Downloads in chunks of BULK_CHUNK_SIZE to avoid yfinance timeouts on large lists.
    """
    if not tickers:
        return {}

    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    # Split into chunks to prevent yfinance from choking on 500+ tickers
    chunks = [tickers[i:i + BULK_CHUNK_SIZE] for i in range(0, len(tickers), BULK_CHUNK_SIZE)]
    if len(chunks) > 1:
        log.info(f"  Splitting {len(tickers)} tickers into {len(chunks)} chunks of ≤{BULK_CHUNK_SIZE}")

    result = {}
    for ci, chunk in enumerate(chunks):
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by='ticker',
                threads=True,
            )
        except Exception as e:
            log.error(f"yfinance bulk download chunk {ci+1}/{len(chunks)} failed: {e}")
            continue

        if raw.empty:
            continue

        # When only 1 ticker in chunk, yfinance returns flat columns (not multi-level)
        if len(chunk) == 1:
            tk = chunk[0]
            try:
                tk_df = raw.dropna(subset=['Close'])
                if len(tk_df) > 0:
                    result[tk] = tk_df
            except (KeyError, TypeError):
                pass
            continue

        for tk in chunk:
            try:
                tk_df = raw[tk].dropna(subset=['Close'])
                if len(tk_df) > 0:
                    result[tk] = tk_df
            except (KeyError, TypeError):
                continue

        if len(chunks) > 1:
            log.info(f"  Chunk {ci+1}/{len(chunks)}: got {sum(1 for t in chunk if t in result)}/{len(chunk)} tickers")

    return result


def get_sector_for_ticker(ticker):
    """Get sector ETF for a ticker. Uses static map, falls back to yfinance info."""
    if ticker in _TICKER_TO_SECTOR:
        return _TICKER_TO_SECTOR[ticker]
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', '')
        return _YF_SECTOR_TO_ETF.get(sector, 'XLK')
    except Exception:
        return 'XLK'


# ── Technical indicator helpers (match HUD JS exactly) ───────────────────────
def compute_ema(closes, period):
    """EMA matching the HUD's computeEMA (standard EWM)."""
    s = pd.Series(closes)
    return s.ewm(span=period, adjust=False).mean().values


def compute_rsi(closes, period=14):
    """RSI matching the HUD's computeRSI."""
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def compute_atr(highs, lows, closes, period=14):
    """ATR matching the HUD's computeATR — returns single scalar (last value)."""
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return float(atr[-1])


# ── Signal builder (shared by Phase 3 and Phase 3b) ─────────────────────────
def _build_signal(tk, tk_df, sec, spy_ret21):
    """
    Apply V5 filter chain to a single ticker's data.
    Returns (signal_dict, scanned_data_dict) or (None, scanned_data_dict).
    """
    closes = tk_df['Close'].values
    highs = tk_df['High'].values
    lows = tk_df['Low'].values
    volumes = tk_df['Volume'].values
    n = len(closes)
    last_close = float(closes[-1])

    # EMA21 filter
    ema21_arr = compute_ema(closes, 21)
    ema21_val = float(ema21_arr[-1])
    if last_close <= ema21_val:
        return None

    # RSI filter
    rsi_arr = compute_rsi(closes, 14)
    rsi_val = float(rsi_arr[-1])
    if np.isnan(rsi_val) or rsi_val < RSI_LOW or rsi_val > RSI_HIGH:
        return None

    # RS₂₁ check
    stock_ret21 = float(closes[-1] / closes[-22] - 1) if n >= 22 else 0
    rs = stock_ret21 - spy_ret21
    if rs < RS_THRESHOLD:
        return None

    # Conviction sizing (cast to native bool — numpy.bool_ breaks Pydantic JSON)
    has_pullback = bool(n >= 5 and
                        (closes[-2] < closes[-3] or
                         closes[-3] < closes[-4] or
                         closes[-4] < closes[-5]))
    vol_slice = volumes[max(0, n - 21):n - 1]
    vol_avg20 = float(np.mean(vol_slice)) if len(vol_slice) > 0 else 1
    vol_ratio = float(volumes[-1] / vol_avg20) if vol_avg20 > 0 else 1
    conviction = bool(has_pullback or vol_ratio >= VOL_CONVICTION)

    # ATR + stops
    atr = compute_atr(highs, lows, closes, 14)
    stop = last_close - ATR_STOP_MULT * atr
    trail_stop = ema21_val - TRAIL_ATR_MULT * atr
    spark = [float(c) for c in closes[-15:]]

    prev_close = float(closes[-2]) if n >= 2 else last_close
    chg = round((last_close / prev_close - 1) * 100, 2)

    return {
        'tk': tk,
        'name': tk,
        'sec': sec,
        'secName': SECTOR_NAMES.get(sec, sec),
        'px': round(last_close, 2),
        'chg': chg,
        'rs21': round(rs * 100, 1),
        'ema21': round(ema21_val, 2),
        'rsi': round(rsi_val),
        'atr': round(atr, 2),
        'rvol': round(vol_ratio, 2),
        'pullback': has_pullback,
        'vol': int(volumes[-1]),
        'spark': spark,
        'conv': 'CONVICTION' if conviction else 'BASE',
        'risk': 2 if conviction else 1,
        'stop': round(stop, 2),
        'trail': round(trail_stop, 2),
        'rPerShare': round(last_close - stop, 2),
    }


# ── Core V5 scan logic ──────────────────────────────────────────────────────
def v5_scan(tickers_by_sector=None, top_n=2, personal_watchlist=None, allow_intraday=True):
    """
    Run the V5 Adaptive Momentum scan.

    Args:
        tickers_by_sector: dict mapping sector ETF → [tickers].
                           Defaults to DEFAULT_UNIVERSE.
        top_n: number of top sectors to allow (default 2).
        personal_watchlist: list of tickers to scan without sector filter.
        allow_intraday: if True, keep today's partial bar during market hours
                        and tag signals as INTRADAY. If False, drop partial bars
                        (legacy behaviour used by auto-scan for confirmed-only).

    Returns:
        dict with keys: regime, sectors, signals, weeklyPerformers,
                        personalSignals, personalStatus, meta
    """
    t0 = time.time()

    if tickers_by_sector is None:
        tickers_by_sector = fetch_sp500_universe()
    if personal_watchlist is None:
        personal_watchlist = []

    # Rebuild ticker→sector reverse map
    _ensure_ticker_map(tickers_by_sector)

    # Date range: ~180 days to cover SECTOR_LOOKBACK(63) + EMA warm-up
    end = datetime.date.today().strftime('%Y-%m-%d')
    start = (datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')

    # Build ticker→sector map
    ticker_sector_map = {}
    for etf, tickers in tickers_by_sector.items():
        for tk in tickers:
            ticker_sector_map[tk] = etf

    # ── PHASE 1: Regime + Sector ranking (1 bulk download) ───────────────
    log.info("Phase 1: Fetching SPY + sector ETFs")
    phase1_tickers = ['SPY'] + SECTOR_ETFS
    phase1_data = fetch_bulk(phase1_tickers, start, end)
    phase1_data, bar_date, is_intraday = _drop_today_if_incomplete(phase1_data, keep_intraday=allow_intraday)
    t1 = round(time.time() - t0, 1)
    log.info(f"  Phase 1 download: {t1}s, got {len(phase1_data)}/{len(phase1_tickers)} tickers (bar: {bar_date}, intraday={is_intraday})")

    spy_df = phase1_data.get('SPY')
    if spy_df is None or len(spy_df) < 50:
        raise RuntimeError("Could not fetch SPY data — check yfinance connectivity")

    spy_closes = spy_df['Close'].values
    spy_ema50 = compute_ema(spy_closes, 50)
    spy_last = float(spy_closes[-1])
    spy_ema50_last = float(spy_ema50[-1])
    regime_bull = spy_last > spy_ema50_last

    regime = {
        'state': 'BULL' if regime_bull else 'BEAR',
        'spyPx': round(spy_last, 2),
        'spyEma50': round(spy_ema50_last, 2),
        'abovePct': round((spy_last / spy_ema50_last - 1) * 100, 1),
        'bull': regime_bull,
    }
    log.info(f"  Regime: {regime['state']} (SPY={spy_last:.2f}, EMA50={spy_ema50_last:.2f})")

    # Sector ranking
    spy_ret63 = float(spy_closes[-1] / spy_closes[-SECTOR_LOOKBACK] - 1) if len(spy_closes) >= SECTOR_LOOKBACK else 0
    sector_ranking = []
    sector_uptrend = {}

    for etf in SECTOR_ETFS:
        etf_df = phase1_data.get(etf)
        if etf_df is None or len(etf_df) < SECTOR_LOOKBACK:
            sector_uptrend[etf] = False
            continue
        ec = etf_df['Close'].values
        etf_ret = float(ec[-1] / ec[-SECTOR_LOOKBACK] - 1)
        e21 = compute_ema(ec, 21)
        uptrend = bool(ec[-1] > e21[-1])
        sector_uptrend[etf] = uptrend
        sector_ranking.append({
            'etf': etf,
            'name': SECTOR_NAMES.get(etf, etf),
            'rs': etf_ret - spy_ret63,
            'uptrend': uptrend,
        })

    sector_ranking.sort(key=lambda s: s['rs'], reverse=True)
    allowed_sectors = set(s['etf'] for s in sector_ranking[:top_n])

    sectors_out = [
        {
            'etf': s['etf'],
            'name': s['name'],
            'rs3m': round(s['rs'] * 100, 1),
            'top': s['etf'] in allowed_sectors,
            'uptrend': s['uptrend'],
        }
        for s in sector_ranking
    ]
    log.info(f"  Top {top_n} sectors: {sorted(allowed_sectors)}")

    # ── PHASE 2: Filter to top sectors ───────────────────────────────────
    sector_filtered = []
    for etf, tickers in tickers_by_sector.items():
        if etf in allowed_sectors and sector_uptrend.get(etf, False):
            sector_filtered.extend(tickers)

    log.info(f"Phase 2: {len(sector_filtered)} stocks in active sectors (from {len(ticker_sector_map)} total)")

    # ── PHASE 3: Bulk download + signal generation (1 bulk download) ────
    signals = []
    scanned_data = {}
    spy_ret21 = float(spy_closes[-1] / spy_closes[-22] - 1) if len(spy_closes) >= 22 else 0
    spy_week_ret = float(spy_closes[-1] / spy_closes[-6] - 1) if len(spy_closes) >= 6 else 0

    if regime_bull and sector_filtered:
        log.info(f"Phase 3: Fetching {len(sector_filtered)} tickers in bulk")
        phase3_data = fetch_bulk(sector_filtered, start, end)
        phase3_data, _, _ = _drop_today_if_incomplete(phase3_data, keep_intraday=allow_intraday)
        t3 = round(time.time() - t0, 1)
        log.info(f"  Phase 3 download: {t3}s total, got {len(phase3_data)}/{len(sector_filtered)} tickers")

        for tk in sector_filtered:
            tk_df = phase3_data.get(tk)
            if tk_df is None or len(tk_df) < 22:
                continue

            closes = tk_df['Close'].values
            n = len(closes)
            last_close = float(closes[-1])
            sec = ticker_sector_map.get(tk, 'XLK')

            # Weekly perf tracking (for weekly performers table)
            week_ret = float(closes[-1] / closes[-6] - 1) if n >= 6 else 0
            scanned_data[tk] = {
                'close': round(last_close, 2),
                'weekRet': round(week_ret * 100, 1),
                'rs5d': round((week_ret - spy_week_ret) * 100, 1),
                'sec': sec,
                'secName': SECTOR_NAMES.get(sec, sec),
            }

            # Apply V5 filter chain
            sig = _build_signal(tk, tk_df, sec, spy_ret21)
            if sig is not None:
                sig['status'] = 'INTRADAY' if is_intraday else 'FRESH'
                signals.append(sig)

        signals.sort(key=lambda s: s['rs21'], reverse=True)
        log.info(f"  Phase 3 signals: {len(signals)} found (status={'INTRADAY' if is_intraday else 'FRESH'})")

    # Weekly performers (top 15 by 5d RS)
    weekly_performers = sorted(
        [
            {
                'tk': tk,
                'close': d['close'],
                'weekRet': d['weekRet'],
                'rs5d': d['rs5d'],
                'sec': d['sec'],
                'secName': d['secName'],
            }
            for tk, d in scanned_data.items()
        ],
        key=lambda x: x['rs5d'],
        reverse=True,
    )[:15]

    # ── PHASE 3b: Personal watchlist (1 bulk download, no sector filter) ─
    personal_signals = []
    personal_status = []

    if personal_watchlist:
        log.info(f"Phase 3b: Personal watchlist ({len(personal_watchlist)} tickers)")
        pw_data = fetch_bulk(personal_watchlist, start, end)
        pw_data, _, _ = _drop_today_if_incomplete(pw_data, keep_intraday=allow_intraday)

        # For large watchlists, skip individual yfinance sector lookups (too slow)
        use_fast_sector = len(personal_watchlist) > 50

        for tk in personal_watchlist:
            if use_fast_sector:
                sec = _TICKER_TO_SECTOR.get(tk, 'XLK')
            else:
                sec = _TICKER_TO_SECTOR.get(tk, get_sector_for_ticker(tk))
            sec_in_top = sec in allowed_sectors
            sec_up = sector_uptrend.get(sec, False)

            status = {
                'tk': tk, 'sec': sec, 'secName': SECTOR_NAMES.get(sec, sec),
                'sectorInUptrend': sec_up, 'inTopSector': sec_in_top,
            }

            tk_df = pw_data.get(tk)
            if tk_df is None or len(tk_df) < 22:
                status['error'] = 'insufficient data'
                personal_status.append(status)
                continue

            closes = tk_df['Close'].values
            n = len(closes)
            last_close = float(closes[-1])

            ema21_arr = compute_ema(closes, 21)
            ema21_val = float(ema21_arr[-1])
            rsi_arr = compute_rsi(closes, 14)
            rsi_val = float(rsi_arr[-1])
            stock_ret21 = float(closes[-1] / closes[-22] - 1) if n >= 22 else 0
            rs = stock_ret21 - spy_ret21

            status['px'] = round(last_close, 2)
            status['ema21'] = round(ema21_val, 2)
            status['rsi'] = round(rsi_val) if not np.isnan(rsi_val) else None
            status['rs21'] = round(rs * 100, 1)
            status['aboveEma21'] = bool(last_close > ema21_val)
            status['rsiInRange'] = bool(RSI_LOW <= rsi_val <= RSI_HIGH) if not np.isnan(rsi_val) else False
            status['rsPass'] = bool(rs >= RS_THRESHOLD)
            personal_status.append(status)

            # Apply V5 filter chain (same as main scan, minus sector filter)
            sig = _build_signal(tk, tk_df, sec, spy_ret21)
            if sig is not None:
                sig['sectorInUptrend'] = sec_up
                sig['inTopSector'] = sec_in_top
                sig['status'] = 'INTRADAY' if is_intraday else 'FRESH'
                personal_signals.append(sig)

    duration = round(time.time() - t0, 1)

    return {
        'timestamp': datetime.datetime.now().isoformat(),
        'date': end,
        'regime': regime,
        'sectors': sectors_out,
        'signals': signals,
        'weeklyPerformers': weekly_performers,
        'personalSignals': personal_signals,
        'personalStatus': personal_status,
        'meta': {
            'universe_size': len(ticker_sector_map),
            'sector_filtered': len(sector_filtered),
            'signals_found': len(signals),
            'personal_signals': len(personal_signals),
            'personal_watchlist': personal_watchlist or [],
            'top_n': top_n,
            'duration_seconds': duration,
            'data_source': 'yfinance',
            'bar_date': bar_date,
            'market_closed': _market_is_closed(),
            'is_intraday': is_intraday,
        },
    }
