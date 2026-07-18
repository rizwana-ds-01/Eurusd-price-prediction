"""
feature_builder.py — v18.7 103-feature builder + X_all(114) assembler
VERIFIED against B_scaled_v6_clean.pkl FEATURES_FINAL (exact 103 names, exact order).

UDL layout in X_all (positions 103-113):
  103=gmm_p0, 104=gmm_p1, 105=gmm_p2, 106=gmm_p3
  107=hmm_p0, 108=hmm_p1, 109=hmm_p2, 110=hmm_p3
  111=gmm_conf, 112=hmm_conf, 113=size_mult
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

FEATURES_FINAL = [
    'is_weekend_gap','is_holiday_gap','post_holiday','day_of_week','has_large_gap',
    'is_forex_holiday','is_post_forex_holiday','is_suspected_holiday','hour',
    'move_pips','is_extreme_50pips','is_extreme_100pips','gap_suspicious',
    'tick_volume_d1','spread_d1','tick_volume_w1','spread_w1','is_high_vol',
    'is_news_event','bos_signal','swing_high','swing_low','demand_ob','supply_ob',
    'demand_ob_strength','supply_ob_strength','at_demand_zone','at_supply_zone',
    'zone_reaction','adx','di_plus','di_minus','rsi','vol_percentile','momentum',
    'is_pullback','pullback_to_ema50_bull','pullback_to_ema50_bear',
    'pullback_depth_bull','pullback_depth_bear','pullback_depth',
    'equal_highs','equal_lows','liquidity_sweep','is_range','range_width',
    'range_break','accumulation','distribution','mtf_bullish_alignment',
    'bos_strength','choch_signal','gap_close','mtf_mixed','trend_strength_ema',
    'macro_bias','swing_signal','price_roc_1','price_roc_3','price_roc_5',
    'momentum_velocity','momentum_acceleration','setup_type_encoded',
    'momentum_regime_scaled','vol_percentile_regime_scaled','atr_percentile',
    'bar_range','range_percentile','session_asia','session_london',
    'session_newyork','session_overlap','session_intensity','state_4h_enc',
    'state_1d_enc','state_1w_enc','structure_state','zone_strength',
    'near_key_zone','upper_wick','lower_wick','body_ratio','candle_dir',
    'consecutive_moves','gap_type_enc','session_enc','trend_enc',
    'trend_strength_enc','vol_regime_enc','trend_maturity_enc',
    'price_vs_ema50','price_vs_ema200','ema20_vs_ema50','ema50_vs_ema200',
    'price_vs_d1','price_vs_w1','macd_norm','macd_sig_norm','macd_hist_norm',
    'atr_14_pips','spread_vs_atr','volume_log','gap_pips_log'
]  # 103 features — exact FEATURES_FINAL from B_scaled_v6_clean.pkl

def build_raw_features(df: pd.DataFrame) -> dict:
    """Build 103 raw features from OHLCV DataFrame (≥200 bars, datetime index)."""
    assert len(df) >= 200, "Need ≥200 bars"
    f = {}
    bar=df.iloc[-1]; prev=df.iloc[-2]; dt=pd.Timestamp(df.index[-1])
    close=float(bar['close']); high=float(bar['high'])
    low=float(bar['low']); open_=float(bar['open'])
    vol=float(bar.get('volume',5000.0)); spr=float(bar.get('spread',1.5))
    ema20=float(df['close'].ewm(span=20,adjust=False).mean().iloc[-1])
    ema50=float(df['close'].ewm(span=50,adjust=False).mean().iloc[-1])
    ema200=float(df['close'].ewm(span=200,adjust=False).mean().iloc[-1])
    tr=pd.concat([df['high']-df['low'],(df['high']-df['close'].shift()).abs(),(df['low']-df['close'].shift()).abs()],axis=1).max(axis=1)
    atr14=max(float(tr.rolling(14).mean().iloc[-1]),1e-8); atr_pips=max(atr14*10000,1.0)
    delta=df['close'].diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rsi=float((100-100/(1+gain/loss.replace(0,1e-10))).iloc[-1]/100.0)
    ema12=df['close'].ewm(span=12,adjust=False).mean(); ema26=df['close'].ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; sig_=macd.ewm(span=9,adjust=False).mean(); hist=macd-sig_
    dm_p=df['high'].diff().clip(lower=0); dm_m=(-df['low'].diff()).clip(lower=0); tr14=tr.rolling(14).mean()
    di_p=float(100*dm_p.rolling(14).mean().iloc[-1]/max(float(tr14.iloc[-1]),1e-8))
    di_m=float(100*dm_m.rolling(14).mean().iloc[-1]/max(float(tr14.iloc[-1]),1e-8))
    adx=float(abs(di_p-di_m)/max(di_p+di_m,1e-8))
    vol_pct=float(df['volume'].rolling(100).rank(pct=True).iloc[-1]) if 'volume' in df.columns else 0.5
    if pd.isna(vol_pct): vol_pct=0.5
    body=abs(close-open_); total=max(high-low,1e-8)
    wick_u=high-max(close,open_); wick_d=min(close,open_)-low
    rh20=float(df['high'].iloc[-20:-1].max()); rl20=float(df['low'].iloc[-20:-1].min())
    bos=1 if (close>rh20 or close<rl20) else 0
    bos_str=float(max((close-rh20)/atr14,(rl20-close)/atr14,0))
    choch=0
    if len(df)>3:
        pd_=float(df['close'].iloc[-3])-float(df['close'].iloc[-4]); cd_=close-float(prev['close'])
        choch=1 if (pd_>0 and cd_<0) or (pd_<0 and cd_>0) else 0
    sw_h=int(len(df)>4 and df['high'].iloc[-3]>df['high'].iloc[-4] and df['high'].iloc[-3]>df['high'].iloc[-2])
    sw_l=int(len(df)>4 and df['low'].iloc[-3]<df['low'].iloc[-4] and df['low'].iloc[-3]<df['low'].iloc[-2])
    rng_w=float((rh20-rl20)/atr14); is_rng=1 if rng_w<3.0 else 0
    rng_br=int(len(df)>25 and (close>float(df['high'].iloc[-25:-5].max()) or close<float(df['low'].iloc[-25:-5].min())))
    mom=float((close/float(df['close'].iloc[-11])-1) if len(df)>11 else 0)
    mom_v=float(close-float(prev['close'])-(float(prev['close'])-float(df['close'].iloc[-3]))) if len(df)>3 else 0.0
    mom_a=0.0
    if len(df)>4:
        v2=float(df['close'].iloc[-2])-float(df['close'].iloc[-3])-(float(df['close'].iloc[-3])-float(df['close'].iloc[-4])); mom_a=mom_v-v2
    roc1=float((close-float(prev['close']))/max(abs(float(prev['close'])),1e-8))
    roc3=float((close-float(df['close'].iloc[-4]))/max(abs(float(df['close'].iloc[-4])),1e-8)) if len(df)>4 else 0.0
    roc5=float((close-float(df['close'].iloc[-6]))/max(abs(float(df['close'].iloc[-6])),1e-8)) if len(df)>6 else 0.0
    h=dt.hour; s_as=int(0<=h<9); s_lo=int(7<=h<16); s_ny=int(12<=h<21); s_ov=int(12<=h<16)
    s_in=1.0 if s_ov else (0.7 if (s_lo or s_ny) else 0.3); s_en=3 if s_ov else (2 if s_ny else (1 if s_lo else 0))
    mtf_bull=1 if (ema50>ema200 and close>ema50 and rsi>0.5) else 0; mtf_mix=1 if ((close>ema200)!=(close>ema50)) else 0
    macro=0.5 if ema50>ema200 else -0.5; ts_ema=float(np.tanh((ema50-ema200)/atr14*5))
    trend_e=2 if ema50>ema200*1.001 else (-2 if ema50<ema200*0.999 else 0)
    ts_enc=3 if adx>0.04 else (2 if adx>0.02 else (1 if adx>0.01 else 0))
    vol_r=2 if vol_pct>0.70 else (0 if vol_pct<0.30 else 1)
    setup_t=3 if (bos and close>ema200) else (2 if abs(close-ema50)<atr14*0.3 else (1 if bos else 0))
    gap_pips=(open_-float(prev['close']))*10000
    gap_t=3 if abs(gap_pips)>atr_pips*2 else (2 if abs(gap_pips)>atr_pips else (1 if abs(gap_pips)>atr_pips*0.5 else 0))
    s_enc2=int(2 if close>ema200+atr14*0.5 else (-2 if close<ema200-atr14*0.5 else (1 if close>ema50 else 0)))
    is_pb=1 if abs(close-ema50)<atr14*0.3 else 0
    pb_bull=float(max(-(close-ema50)/atr14,0)); pb_bear=float(max((close-ema50)/atr14,0))
    pb_db=float(max(rh20-close,0)/atr14); pb_dbe=float(max(close-rl20,0)/atr14)
    at_dem=int(abs(close-rl20)<atr14*0.5); at_sup=int(abs(close-rh20)<atr14*0.5)
    dem_ob=int(float(prev['close'])<float(prev['open']) and close>float(df['high'].iloc[-2]))
    sup_ob=int(float(prev['close'])>float(prev['open']) and close<float(df['low'].iloc[-2]))
    eq_h=int(df['high'].rolling(10).apply(lambda x:(x.diff().abs()<atr14*0.1).sum()>=2).iloc[-1])
    eq_l=int(df['low'].rolling(10).apply(lambda x:(x.diff().abs()<atr14*0.1).sum()>=2).iloc[-1])
    liq=int((high>float(df['high'].iloc[-10:-1].max()) and close<float(df['high'].iloc[-10:-1].max())) or
            (low<float(df['low'].iloc[-10:-1].min()) and close>float(df['low'].iloc[-10:-1].min())))
    accum=int(is_rng and len(df)>20 and df['volume'].iloc[-5:].mean()>df['volume'].iloc[-20:].mean()*1.2)
    distr=int(close>float(df['high'].rolling(20).quantile(0.8).iloc[-1]) and len(df)>20 and df['volume'].iloc[-5:].mean()>df['volume'].iloc[-20:].mean()*1.2)
    atr_pct=float(tr.rolling(100).rank(pct=True).iloc[-1]); atr_pct=atr_pct if not pd.isna(atr_pct) else 0.5
    rng_pct=float((df['high']-df['low']).rolling(100).rank(pct=True).iloc[-1]); rng_pct=rng_pct if not pd.isna(rng_pct) else 0.5
    gap_cl=int(gap_pips!=0 and (close-open_)*gap_pips<0)
    tm_enc=1 if bos else (-1 if choch else 0); z_str=float(bos_str+(0.5 if dem_ob or sup_ob else 0))
    z_react=float((close-float(prev['close']))/atr14); sw_sig=int(bos or choch)
    p_d1=float((close-float(prev['close']))/atr14)
    p_w1=float((close-float(df['close'].iloc[-5]))/atr14) if len(df)>5 else 0.0
    has_lg=int(abs(gap_pips)>atr_pips*2)
    consec=0
    if len(df)>6:
        dirs=[1 if float(df['close'].iloc[-i])>float(df['close'].iloc[-i-1]) else -1 for i in range(1,6)]
        d0=dirs[0]; consec=d0*sum(1 for d in dirs if d==d0)
    f['is_weekend_gap']=1 if dt.dayofweek==0 and h<4 else 0; f['is_holiday_gap']=0; f['post_holiday']=0
    f['day_of_week']=int(dt.dayofweek); f['has_large_gap']=has_lg
    f['is_forex_holiday']=0; f['is_post_forex_holiday']=0; f['is_suspected_holiday']=0
    f['hour']=int(h); f['move_pips']=float(gap_pips)
    f['is_extreme_50pips']=int(abs(gap_pips)>50); f['is_extreme_100pips']=int(abs(gap_pips)>100)
    f['gap_suspicious']=int(abs(gap_pips)>atr_pips*3)
    f['tick_volume_d1']=float(vol); f['spread_d1']=float(spr)
    f['tick_volume_w1']=float(df['volume'].iloc[-5:].sum()) if len(df)>5 else float(vol)
    f['spread_w1']=float(df['spread'].iloc[-5:].mean()) if ('spread' in df.columns and len(df)>5) else float(spr)
    f['is_high_vol']=int(vol_pct>0.70); f['is_news_event']=int(bar.get('is_news_event',0))
    f['bos_signal']=bos; f['swing_high']=sw_h; f['swing_low']=sw_l
    f['demand_ob']=dem_ob; f['supply_ob']=sup_ob
    f['demand_ob_strength']=bos_str if dem_ob else 0.0; f['supply_ob_strength']=bos_str if sup_ob else 0.0
    f['at_demand_zone']=at_dem; f['at_supply_zone']=at_sup; f['zone_reaction']=z_react
    f['adx']=adx; f['di_plus']=di_p/100.0; f['di_minus']=di_m/100.0
    f['rsi']=rsi; f['vol_percentile']=vol_pct; f['momentum']=mom
    f['is_pullback']=is_pb; f['pullback_to_ema50_bull']=pb_bull; f['pullback_to_ema50_bear']=pb_bear
    f['pullback_depth_bull']=pb_db; f['pullback_depth_bear']=pb_dbe; f['pullback_depth']=max(pb_db,pb_dbe)
    f['equal_highs']=eq_h; f['equal_lows']=eq_l; f['liquidity_sweep']=liq
    f['is_range']=is_rng; f['range_width']=rng_w; f['range_break']=rng_br
    f['accumulation']=accum; f['distribution']=distr; f['mtf_bullish_alignment']=mtf_bull
    f['bos_strength']=bos_str; f['choch_signal']=choch; f['gap_close']=gap_cl
    f['mtf_mixed']=mtf_mix; f['trend_strength_ema']=ts_ema; f['macro_bias']=macro
    f['swing_signal']=sw_sig; f['price_roc_1']=roc1; f['price_roc_3']=roc3; f['price_roc_5']=roc5
    f['momentum_velocity']=mom_v; f['momentum_acceleration']=mom_a; f['setup_type_encoded']=setup_t
    f['momentum_regime_scaled']=float(mom*vol_pct); f['vol_percentile_regime_scaled']=float(vol_pct*adx)
    f['atr_percentile']=atr_pct; f['bar_range']=float((high-low)*10000); f['range_percentile']=rng_pct
    f['session_asia']=s_as; f['session_london']=s_lo; f['session_newyork']=s_ny
    f['session_overlap']=s_ov; f['session_intensity']=s_in
    f['state_4h_enc']=s_enc2; f['state_1d_enc']=s_enc2; f['state_1w_enc']=s_enc2
    f['structure_state']=int(2 if (bos and close>ema200) else (-2 if (bos and close<ema200) else (1 if close>ema200 else -1)))
    f['zone_strength']=z_str; f['near_key_zone']=int(at_dem or at_sup)
    f['upper_wick']=float(wick_u/total); f['lower_wick']=float(wick_d/total)
    f['body_ratio']=float(body/total); f['candle_dir']=1 if close>open_ else -1
    f['consecutive_moves']=consec; f['gap_type_enc']=gap_t; f['session_enc']=s_en
    f['trend_enc']=trend_e; f['trend_strength_enc']=ts_enc; f['vol_regime_enc']=vol_r
    f['trend_maturity_enc']=tm_enc
    f['price_vs_ema50']=float((close-ema50)/atr14); f['price_vs_ema200']=float((close-ema200)/atr14)
    f['ema20_vs_ema50']=float((ema20-ema50)/atr14); f['ema50_vs_ema200']=float((ema50-ema200)/atr14)
    f['price_vs_d1']=p_d1; f['price_vs_w1']=p_w1
    f['macd_norm']=float(float(macd.iloc[-1])/atr14); f['macd_sig_norm']=float(float(sig_.iloc[-1])/atr14)
    f['macd_hist_norm']=float(float(hist.iloc[-1])/atr14)
    f['atr_14_pips']=atr_pips; f['spread_vs_atr']=float(spr/max(atr_pips,0.01))
    f['volume_log']=float(np.log1p(vol)); f['gap_pips_log']=float(np.sign(gap_pips)*np.log1p(abs(gap_pips)))
    return f

def features_to_vector(features: dict) -> np.ndarray:
    return np.array([float(features.get(k,0.0)) for k in FEATURES_FINAL], dtype=np.float64)

def get_ema20_close(df):
    """Return (ema20, close) for pullback filter."""
    close = float(df['close'].iloc[-1])
    ema20 = float(df['close'].ewm(span=20, adjust=False).mean().iloc[-1])
    return ema20, close

def build_full_matrix(raw_vec, b_scaler, gmm_probs, hmm_probs, size_mult, top_k_idx):
    """raw(103) → scale → +UDL(11) → X_all(114) → top_k → X_ml(1,64)"""
    assert raw_vec.shape==(103,)
    X_sc=b_scaler.transform(raw_vec.reshape(1,-1)); assert X_sc.shape==(1,103)
    udl=np.array([gmm_probs[0],gmm_probs[1],gmm_probs[2],gmm_probs[3],hmm_probs[0],hmm_probs[1],hmm_probs[2],hmm_probs[3],float(gmm_probs.max()),float(hmm_probs.max()),float(size_mult)])
    X_full=np.hstack([X_sc,udl.reshape(1,-1)]); assert X_full.shape==(1,114)
    X_ml=X_full[:,top_k_idx]; assert X_ml.shape==(1,64)
    return X_ml