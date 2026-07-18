"""trade_manager.py — unchanged from verified v18.7 version"""
import logging, json
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    TP1='TP1_HIT'; TP2='TP2_HIT'; SL='SL_HIT'; BE='BE_HIT'; TIME='TIME_EXIT'

@dataclass
class OpenTrade:
    trade_id:str; symbol:str; direction:int
    entry_price:float; entry_time:datetime
    sl_price:float; tp1_price:float; runner_tp:float
    sl_pips:float; tp_pips:float
    lot_size:float; tp1_lots:float; runner_lots:float
    atr_pips:float; spread_pips:float; slippage_pips:float
    regime:int; fold_quality:str; expected_edge:float; signal_prob:float
    bars_open:int=0; tp1_closed:bool=False
    runner_sl:Optional[float]=None; realized_pnl:float=0.0; status:Optional[str]=None

class TradeManager:
    REGIME_NAMES={0:'BULL_TREND',1:'BEAR_TREND',2:'RANGE_CHOP',3:'ANOMALY'}
    def __init__(self,cfg):
        self.max_bars=cfg.get('max_hold_bars',20); self.tp1_frac=cfg.get('tp1_fraction',0.70)
        self.run_frac=cfg.get('runner_fraction',0.30); self.icm=cfg.get('icm_cost_fixed',1.20)
        self.trail_mult=cfg.get('trail_atr_mult',3.0); self.max_conc=cfg.get('max_concurrent',3)
        self.risk_pct=cfg.get('risk_per_trade',0.01); self.pip_value=cfg.get('pip_value',10.0)
        self.pip_size=cfg.get('pip_size',0.0001); self.slip_norm=cfg.get('slippage_pips_normal',0.5)
        self.slip_news=cfg.get('slippage_pips_news',2.0); self.daily_lim=cfg.get('daily_loss_limit_pct',0.03)
        self._daily_loss=0.0; self._reset_day:Optional[date]=None
        self.open_trades:Dict[str,OpenTrade]={}; self.closed_trades:List[dict]=[]
    def _reset(self):
        today=date.today()
        if self._reset_day!=today: self._daily_loss=0.0; self._reset_day=today
    def _circuit_ok(self,balance):
        self._reset(); return self._daily_loss<self.daily_lim*balance
    def _lots(self,balance,sl_pips):
        return max(0.01,round(balance*self.risk_pct/max(sl_pips*self.pip_value,1),2))
    def open_trade(self,signal,entry_price,balance,symbol='EURUSD',is_news=False):
        self._reset()
        if not self._circuit_ok(balance): logger.warning("CIRCUIT BREAKER"); return None
        if len(self.open_trades)>=self.max_conc: logger.warning("Max concurrent reached"); return None
        slip=self.slip_news if is_news else self.slip_norm
        d=1 if signal.direction=='BUY' else -1
        eff=entry_price+d*slip*self.pip_size
        sl_p=eff-d*signal.sl_pips*self.pip_size; tp1_p=eff+d*signal.tp1_pips*self.pip_size
        run_p=eff+d*signal.runner_pips*self.pip_size
        lots=self._lots(balance,signal.sl_pips)
        tid=f"{symbol}_{signal.direction}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        t=OpenTrade(trade_id=tid,symbol=symbol,direction=d,
            entry_price=eff,entry_time=datetime.now(timezone.utc),
            sl_price=sl_p,tp1_price=tp1_p,runner_tp=run_p,
            sl_pips=signal.sl_pips,tp_pips=signal.tp_pips,
            lot_size=lots,tp1_lots=round(lots*self.tp1_frac,2),
            runner_lots=round(lots*self.run_frac,2),
            atr_pips=signal.atr_pips,spread_pips=signal.spread_pips,slippage_pips=slip,
            regime=signal.regime,fold_quality=signal.gate_rejected,
            expected_edge=signal.expected_edge,signal_prob=max(signal.cat_buy_prob,signal.cat_sell_prob))
        self.open_trades[tid]=t
        logger.info(f"OPEN|{tid}|{signal.direction}|entry={eff:.5f}|SL={sl_p:.5f}({signal.sl_pips:.0f}pip)|TP1={tp1_p:.5f}|lots={lots:.2f}|slip={slip:.1f}pip")
        return t
    def update(self,trade_id,price,balance=10000.0):
        if trade_id not in self.open_trades: return {'action':'none'}
        t=self.open_trades[trade_id]; t.bars_open+=1
        pnl_pips=t.direction*(price-t.entry_price)/self.pip_size
        if t.bars_open>=self.max_bars:
            t.realized_pnl=pnl_pips-self.icm-t.slippage_pips; t.status=ExitReason.TIME.value
            self._reg_loss(t.realized_pnl,balance); self._close(t,price)
            logger.info(f"TIME_EXIT|{trade_id}|pnl={t.realized_pnl:.1f}pip"); return {'action':'close_all','pnl_pips':t.realized_pnl,'reason':'TIME_EXIT'}
        sl_c=t.runner_sl if (t.tp1_closed and t.runner_sl is not None) else t.sl_price
        sl_hit=(t.direction==1 and price<=sl_c) or (t.direction==-1 and price>=sl_c)
        if sl_hit:
            if t.tp1_closed: t.realized_pnl+=-(self.icm+t.slippage_pips); t.status=ExitReason.BE.value
            else: t.realized_pnl=-(t.sl_pips+self.icm+t.slippage_pips); t.status=ExitReason.SL.value
            self._reg_loss(t.realized_pnl,balance); self._close(t,price)
            logger.info(f"{t.status}|{trade_id}|pnl={t.realized_pnl:.1f}pip"); return {'action':'close_all','pnl_pips':t.realized_pnl,'reason':t.status}
        tp1_hit=(not t.tp1_closed and ((t.direction==1 and price>=t.tp1_price) or (t.direction==-1 and price<=t.tp1_price)))
        if tp1_hit:
            tp1_pnl=t.tp_pips*self.tp1_frac-(self.icm+t.slippage_pips); t.realized_pnl+=tp1_pnl
            t.tp1_closed=True; t.runner_sl=t.entry_price; t.status=ExitReason.TP1.value
            logger.info(f"TP1|{trade_id}|pnl={tp1_pnl:.1f}pip|SL→BE"); return {'action':'close_tp1','pnl_pips':tp1_pnl,'close_lots':t.tp1_lots}
        if self.trail_mult>0 and t.tp1_closed and t.runner_sl is not None:
            trail=t.atr_pips*self.trail_mult*self.pip_size
            new_sl=price-t.direction*trail
            if t.direction==1 and new_sl>t.runner_sl: t.runner_sl=new_sl
            elif t.direction==-1 and new_sl<t.runner_sl: t.runner_sl=new_sl
        if t.tp1_closed:
            tp2_hit=((t.direction==1 and price>=t.runner_tp) or (t.direction==-1 and price<=t.runner_tp))
            if tp2_hit:
                run_pnl=(t.tp_pips+t.atr_pips*4.0)*self.run_frac-(self.icm+t.slippage_pips)
                t.realized_pnl+=run_pnl; t.status=ExitReason.TP2.value; self._close(t,price)
                logger.info(f"TP2|{trade_id}|runner={run_pnl:.1f}pip|total={t.realized_pnl:.1f}pip"); return {'action':'close_all','pnl_pips':t.realized_pnl,'reason':'TP2_HIT'}
        return {'action':'none','bars_open':t.bars_open,'pnl_unrealized':pnl_pips}
    def _reg_loss(self,pnl,balance):
        if pnl<0: self._daily_loss+=abs(pnl*self.pip_size*self.pip_value)/max(balance,1)
    def _close(self,t,price):
        self.closed_trades.append({'trade_id':t.trade_id,'symbol':t.symbol,
            'direction':'BUY' if t.direction==1 else 'SELL','entry':round(t.entry_price,5),
            'close':round(price,5),'sl_pips':round(t.sl_pips,1),'tp_pips':round(t.tp_pips,1),
            'bars':t.bars_open,'lots':t.lot_size,'pnl_pips':round(t.realized_pnl,2),
            'slippage':round(t.slippage_pips,2),'reason':t.status,
            'regime':self.REGIME_NAMES.get(t.regime,'?'),'quality':t.fold_quality,
            'edge':round(t.expected_edge,1),'signal_prob':round(t.signal_prob,4),
            'entry_time':t.entry_time.isoformat(),'close_time':datetime.now(timezone.utc).isoformat()})
        del self.open_trades[t.trade_id]
    def summary(self):
        self._reset()
        if not self.closed_trades: return {'trades':0,'daily_loss_pct':round(self._daily_loss*100,2),'open':len(self.open_trades)}
        pnls=[t['pnl_pips'] for t in self.closed_trades]
        wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<=0]
        return {'trades':len(pnls),'win_rate':round(len(wins)/len(pnls)*100,1),
                'avg_ppt':round(sum(pnls)/len(pnls),1),'total_pnl':round(sum(pnls),1),
                'pf':round(sum(wins)/abs(sum(loss)),2) if loss and sum(loss)!=0 else 999,
                'open':len(self.open_trades),'daily_loss_pct':round(self._daily_loss*100,2)}