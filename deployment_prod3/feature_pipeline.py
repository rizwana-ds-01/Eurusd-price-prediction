"""
feature_pipeline.py  —  EUR/USD H4  v18.14
Builds all 103 features expected by B_scaled_v6_clean scaler.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

FOREX_HOLIDAYS = [
    "2024-01-01","2024-12-25","2024-12-26","2024-01-15","2024-07-04",
    "2025-01-01","2025-12-25","2025-12-26","2025-01-20","2025-07-04",
    "2026-01-01","2026-12-25","2026-12-26",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _safe(arr, fill=0.0):
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(series, fast=12, slow=26, sig=9):
    ema_f = _ema(series, fast)
    ema_s = _ema(series, slow)
    macd  = ema_f - ema_s
    signal= _ema(macd, sig)
    hist  = macd - signal
    return macd, signal, hist


def _percentile_rank(series, window=100):
    return series.rolling(window).rank(pct=True)


def _swing_highs(high, order=3):
    sh = pd.Series(0, index=high.index)
    for i in range(order, len(high) - order):
        window = high.iloc[i-order:i+order+1]
        if high.iloc[i] == window.max():
            sh.iloc[i] = 1
    return sh


def _swing_lows(low, order=3):
    sl = pd.Series(0, index=low.index)
    for i in range(order, len(low) - order):
        window = low.iloc[i-order:i+order+1]
        if low.iloc[i] == window.min():
            sl.iloc[i] = 1
    return sl


# ─────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw OHLCV dataframe with columns
            [time, open, high, low, close, tick_volume, spread]
    Output: DataFrame with exactly 103 features (matching FEATURES_FINAL)
    """

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']
    volume = df['tick_volume'].replace(0, np.nan).ffill().fillna(1)
    spread = df['spread'] if 'spread' in df.columns else pd.Series(0, index=df.index)

    dt = df['time']
    n  = len(df)

    feat = pd.DataFrame(index=df.index)

    # ── Time / calendar ──────────────────────
    feat['is_weekend_gap']       = (dt.dt.dayofweek == 0).astype(int)
    feat['is_holiday_gap']       = dt.dt.strftime('%Y-%m-%d').isin(FOREX_HOLIDAYS).astype(int)
    feat['post_holiday']         = feat['is_holiday_gap'].shift(1).fillna(0)
    feat['day_of_week']          = dt.dt.dayofweek
    gap_pips = (open_ - close.shift(1)).abs() * 10000
    feat['has_large_gap']        = (gap_pips > 20).astype(int)
    feat['is_forex_holiday']     = feat['is_holiday_gap']
    feat['is_post_forex_holiday']= feat['post_holiday']
    feat['is_suspected_holiday'] = ((volume < volume.quantile(0.05)) & (gap_pips > 5)).astype(int)
    feat['hour']                 = dt.dt.hour

    # ── Price action basics ───────────────────
    feat['move_pips']        = (close - open_).abs() * 10000
    feat['is_extreme_50pips']= (feat['move_pips'] > 50).astype(int)
    feat['is_extreme_100pips']=(feat['move_pips'] > 100).astype(int)
    feat['gap_suspicious']   = (gap_pips > 30).astype(int)

    # ── Volume / spread aggregates ────────────
    vol_d1   = volume.rolling(6).sum()
    spread_d1= spread.rolling(6).mean()
    vol_w1   = volume.rolling(30).sum()
    spread_w1= spread.rolling(30).mean()
    feat['tick_volume_d1'] = vol_d1
    feat['spread_d1']      = spread_d1
    feat['tick_volume_w1'] = vol_w1
    feat['spread_w1']      = spread_w1
    feat['is_high_vol']    = (vol_d1 > vol_d1.rolling(20).mean() * 1.5).astype(int)
    feat['is_news_event']  = (feat['move_pips'] > feat['move_pips'].rolling(20).mean() * 2).astype(int)

    # ── Market structure ─────────────────────
    sh = _swing_highs(high)
    sl = _swing_lows(low)
    feat['swing_high'] = sh
    feat['swing_low']  = sl

    # Break of Structure
    prev_sh = high[sh == 1].reindex(high.index).ffill()
    prev_sl = low[sl  == 1].reindex(low.index).ffill()
    feat['bos_signal']    = ((close > prev_sh.shift(1)) | (close < prev_sl.shift(1))).astype(int)
    feat['bos_strength']  = ((close - prev_sh.shift(1)).abs() * 10000).clip(0, 200) / 200
    feat['choch_signal']  = (feat['bos_signal'] & (feat['bos_signal'].shift(1) == 0)).astype(int)

    # Order blocks (demand / supply)
    bearish_ob = (close.shift(1) > open_.shift(1)) & (close < open_)
    bullish_ob = (close.shift(1) < open_.shift(1)) & (close > open_)
    feat['demand_ob']         = bullish_ob.astype(int)
    feat['supply_ob']         = bearish_ob.astype(int)
    feat['demand_ob_strength']= (close - low).abs() * 10000 * bullish_ob
    feat['supply_ob_strength']= (high - close).abs() * 10000 * bearish_ob
    feat['at_demand_zone']    = ((close - low.rolling(20).min()) < 0.0010).astype(int)
    feat['at_supply_zone']    = ((high.rolling(20).max() - close) < 0.0010).astype(int)
    feat['zone_reaction']     = feat['at_demand_zone'] | feat['at_supply_zone']
    feat['zone_strength']     = (feat['demand_ob_strength'] + feat['supply_ob_strength']).clip(0, 300) / 300
    feat['near_key_zone']     = ((feat['at_demand_zone'] + feat['at_supply_zone']) > 0).astype(int)

    # ── Indicators ───────────────────────────
    atr14 = _atr(df, 14)
    rsi14 = _rsi(close, 14)
    adx_window = 14
    pos_dm = (high.diff()).clip(lower=0)
    neg_dm = (-low.diff()).clip(lower=0)
    tr_smooth = atr14 * adx_window
    di_plus  = 100 * pos_dm.rolling(adx_window).mean() / (atr14 + 1e-9)
    di_minus = 100 * neg_dm.rolling(adx_window).mean() / (atr14 + 1e-9)
    dx       = (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9) * 100
    adx      = dx.rolling(adx_window).mean()

    feat['adx']           = adx
    feat['di_plus']       = di_plus
    feat['di_minus']      = di_minus
    feat['rsi']           = rsi14
    feat['vol_percentile']= _percentile_rank(volume, 100)
    feat['momentum']      = close.pct_change(4) * 100

    # ── Pullback / trend ─────────────────────
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200)
    feat['is_pullback']          = ((close < ema20) & (ema20 > ema50)).astype(int) | \
                                   ((close > ema20) & (ema20 < ema50)).astype(int)
    feat['pullback_to_ema50_bull']= ((close > ema50) & (close < ema20)).astype(int)
    feat['pullback_to_ema50_bear']= ((close < ema50) & (close > ema20)).astype(int)
    feat['pullback_depth_bull']   = ((ema20 - close) / (atr14 + 1e-9)).clip(0, 5)
    feat['pullback_depth_bear']   = ((close - ema20) / (atr14 + 1e-9)).clip(0, 5)
    feat['pullback_depth']        = (feat['pullback_depth_bull'] + feat['pullback_depth_bear'])

    # ── Liquidity / equal levels ─────────────
    eq_tol = 0.0005
    feat['equal_highs']      = (high.diff().abs() < eq_tol).astype(int)
    feat['equal_lows']       = (low.diff().abs()  < eq_tol).astype(int)
    prev_high = high.rolling(20).max().shift(1)
    feat['liquidity_sweep']  = ((high > prev_high) & (close < prev_high)).astype(int)

    # ── Range / accumulation / distribution ──
    rng_high = high.rolling(20).max()
    rng_low  = low.rolling(20).min()
    rng_width= (rng_high - rng_low) * 10000
    feat['is_range']     = (rng_width < 50).astype(int)
    feat['range_width']  = rng_width
    feat['range_break']  = ((close > rng_high.shift(1)) | (close < rng_low.shift(1))).astype(int)
    feat['accumulation'] = ((rng_width < 30) & (volume < volume.rolling(20).mean())).astype(int)
    feat['distribution'] = ((rng_width < 30) & (volume > volume.rolling(20).mean())).astype(int)

    # ── Multi-timeframe alignment ─────────────
    bull_align = (ema20 > ema50) & (ema50 > ema200)
    bear_align = (ema20 < ema50) & (ema50 < ema200)
    feat['mtf_bullish_alignment'] = bull_align.astype(int)
    feat['mtf_mixed']             = (~bull_align & ~bear_align).astype(int)

    # ── Gap / structure ───────────────────────
    feat['gap_close']    = (gap_pips.rolling(5).sum() < 2).astype(int)
    feat['trend_strength_ema'] = ((ema20 - ema200) * 10000).clip(-200, 200) / 200
    feat['macro_bias']   = np.sign(ema50 - ema200)
    feat['swing_signal'] = (sh | sl).astype(int)

    # ── Momentum ─────────────────────────────
    feat['price_roc_1']           = close.pct_change(1) * 100
    feat['price_roc_3']           = close.pct_change(3) * 100
    feat['price_roc_5']           = close.pct_change(5) * 100
    feat['momentum_velocity']     = feat['price_roc_1'].diff()
    feat['momentum_acceleration'] = feat['momentum_velocity'].diff()

    # ── Encoded categoricals ──────────────────
    # setup type: 0=no setup, 1=BOS, 2=OB, 3=range
    setup = pd.Series(0, index=df.index)
    setup[feat['bos_signal']==1] = 1
    setup[(feat['demand_ob']==1)|(feat['supply_ob']==1)] = 2
    setup[feat['is_range']==1] = 3
    feat['setup_type_encoded'] = setup

    feat['momentum_regime_scaled']  = (feat['momentum'] / 2).clip(-5, 5)
    feat['vol_percentile_regime_scaled'] = feat['vol_percentile'] * 2 - 1

    # ── ATR / bar metrics ─────────────────────
    feat['atr_percentile'] = _percentile_rank(atr14, 100)
    feat['bar_range']      = (high - low) * 10000
    feat['range_percentile']= _percentile_rank(feat['bar_range'], 100)

    # ── Session ───────────────────────────────
    hr = dt.dt.hour
    feat['session_asia']    = ((hr >= 0)  & (hr < 8)).astype(int)
    feat['session_london']  = ((hr >= 8)  & (hr < 13)).astype(int)
    feat['session_newyork'] = ((hr >= 13) & (hr < 22)).astype(int)
    feat['session_overlap'] = ((hr >= 13) & (hr < 17)).astype(int)
    feat['session_intensity']= feat['session_overlap'] * 2 + feat['session_london'] + feat['session_newyork']

    # ── State encodings ───────────────────────
    trend_4h = np.where(ema20 > ema50, 1, np.where(ema20 < ema50, -1, 0))
    trend_1d  = np.where(ema50 > ema200, 1, np.where(ema50 < ema200, -1, 0))
    trend_1w  = trend_1d  # approximate
    feat['state_4h_enc']    = trend_4h
    feat['state_1d_enc']    = trend_1d
    feat['state_1w_enc']    = trend_1w
    feat['structure_state'] = trend_4h + trend_1d  # -2 to 2

    # ── Candle features ───────────────────────
    body   = (close - open_).abs()
    candle_rng = high - low + 1e-9
    feat['upper_wick']       = (high - pd.concat([close, open_], axis=1).max(axis=1)) / candle_rng
    feat['lower_wick']       = (pd.concat([close, open_], axis=1).min(axis=1) - low) / candle_rng
    feat['body_ratio']       = body / candle_rng
    feat['candle_dir']       = np.sign(close - open_)
    feat['consecutive_moves']= np.sign(close - open_).rolling(3).sum()

    # ── More encoded ──────────────────────────
    gap_type = np.where(gap_pips > 30, 2, np.where(gap_pips > 10, 1, 0))
    feat['gap_type_enc']       = gap_type
    session_enc = np.where(feat['session_overlap']==1, 3,
                  np.where(feat['session_newyork']==1, 2,
                  np.where(feat['session_london']==1, 1, 0)))
    feat['session_enc']        = session_enc

    trend_enc = np.where(bull_align, 2, np.where(bear_align, 0, 1))
    feat['trend_enc']          = trend_enc
    feat['trend_strength_enc'] = adx.clip(0, 60) / 60
    vol_regime_enc = np.where(feat['vol_percentile'] > 0.8, 2,
                     np.where(feat['vol_percentile'] < 0.3, 0, 1))
    feat['vol_regime_enc']     = vol_regime_enc
    maturity = close.rolling(50).apply(lambda x: np.argmax(x) / 50)
    feat['trend_maturity_enc'] = maturity

    # ── Price vs MAs ─────────────────────────
    feat['price_vs_ema50']   = (close - ema50)  * 10000
    feat['price_vs_ema200']  = (close - ema200) * 10000
    feat['ema20_vs_ema50']   = (ema20 - ema50)  * 10000
    feat['ema50_vs_ema200']  = (ema50 - ema200) * 10000

    close_d1 = close.rolling(6).mean()
    close_w1 = close.rolling(30).mean()
    feat['price_vs_d1']      = (close - close_d1) * 10000
    feat['price_vs_w1']      = (close - close_w1) * 10000

    # ── MACD ─────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(close)
    atr_norm = atr14 * 10000 + 1e-9
    feat['macd_norm']     = (macd_line * 10000) / atr_norm
    feat['macd_sig_norm'] = (macd_sig  * 10000) / atr_norm
    feat['macd_hist_norm']= (macd_hist * 10000) / atr_norm

    # ── ATR pips / spread vs ATR ─────────────
    feat['atr_14_pips']   = atr14 * 10000
    feat['spread_vs_atr'] = (spread * 10000) / (atr14 * 10000 + 1e-9)
    feat['volume_log']    = np.log1p(volume)
    feat['gap_pips_log']  = np.log1p(gap_pips)

    # ── Enforce exact column order ────────────
    FEATURES_FINAL = [
        'is_weekend_gap','is_holiday_gap','post_holiday','day_of_week','has_large_gap',
        'is_forex_holiday','is_post_forex_holiday','is_suspected_holiday','hour','move_pips',
        'is_extreme_50pips','is_extreme_100pips','gap_suspicious','tick_volume_d1','spread_d1',
        'tick_volume_w1','spread_w1','is_high_vol','is_news_event','bos_signal','swing_high',
        'swing_low','demand_ob','supply_ob','demand_ob_strength','supply_ob_strength',
        'at_demand_zone','at_supply_zone','zone_reaction','adx','di_plus','di_minus','rsi',
        'vol_percentile','momentum','is_pullback','pullback_to_ema50_bull','pullback_to_ema50_bear',
        'pullback_depth_bull','pullback_depth_bear','pullback_depth','equal_highs','equal_lows',
        'liquidity_sweep','is_range','range_width','range_break','accumulation','distribution',
        'mtf_bullish_alignment','bos_strength','choch_signal','gap_close','mtf_mixed',
        'trend_strength_ema','macro_bias','swing_signal','price_roc_1','price_roc_3','price_roc_5',
        'momentum_velocity','momentum_acceleration','setup_type_encoded','momentum_regime_scaled',
        'vol_percentile_regime_scaled','atr_percentile','bar_range','range_percentile',
        'session_asia','session_london','session_newyork','session_overlap','session_intensity',
        'state_4h_enc','state_1d_enc','state_1w_enc','structure_state','zone_strength',
        'near_key_zone','upper_wick','lower_wick','body_ratio','candle_dir','consecutive_moves',
        'gap_type_enc','session_enc','trend_enc','trend_strength_enc','vol_regime_enc',
        'trend_maturity_enc','price_vs_ema50','price_vs_ema200','ema20_vs_ema50','ema50_vs_ema200',
        'price_vs_d1','price_vs_w1','macd_norm','macd_sig_norm','macd_hist_norm','atr_14_pips',
        'spread_vs_atr','volume_log','gap_pips_log'
    ]

    feat = feat[FEATURES_FINAL]
    feat = feat.fillna(0).replace([np.inf, -np.inf], 0)
    return feat
