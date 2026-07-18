"""
icm_broker.py — v18.7 ICM/MT5 Connector
Fixed: proper error codes, auto-reconnect, clear diagnostics.

MT5 Error Codes:
  -6  = Terminal not running / not logged in yet
  10004 = IPC timeout (terminal not open)
  10006 = No connection (not logged in / no internet)
  10013 = Wrong login/password
"""
import os, sys, logging, time
from typing import Optional, List
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logger = logging.getLogger(__name__)


class ICMBroker:
    """ICM Markets via MetaTrader5 Python API."""

    SERVERS = {
        'demo': 'ICMarketsSC-Demo',
        'live': 'ICMarketsSC-Live01',
    }

    def __init__(self, login: int, password: str,
                 demo: bool = True, symbol: str = 'EURUSD'):
        self.login      = login
        self.password   = password
        self.server     = self.SERVERS['demo'] if demo else self.SERVERS['live']
        self.symbol     = symbol
        self.magic      = 187
        self.is_demo    = demo
        self._mt5       = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to MT5 terminal and login.
        IMPORTANT: Open MT5 terminal FIRST and wait for it to show
        'Connected' (green dot) before calling this.
        """
        # Step 1: Import
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error(
                "MetaTrader5 library not installed.\n"
                "Run: pip install MetaTrader5>=5.0.4682")
            return False

        # Step 2: Initialize terminal connection
        logger.info("Connecting to MT5 terminal...")
        if not self._mt5.initialize():
            code, msg = self._mt5.last_error()
            logger.error(
                f"mt5.initialize() failed: code={code} msg='{msg}'\n"
                f"\n  DIAGNOSIS:\n"
                f"  code=-6 or 10004 → MT5 terminal not open\n"
                f"  code=10006       → MT5 not logged in / no internet\n"
                f"\n  FIX:\n"
                f"  1. Open MetaTrader 5 terminal\n"
                f"  2. Login to your ICM account inside MT5\n"
                f"  3. Wait for green connection dot (bottom right)\n"
                f"  4. THEN run this script")
            return False

        # Step 3: Login
        logger.info(f"Logging in: account={self.login} server={self.server}")
        ok = self._mt5.login(self.login, password=self.password, server=self.server)
        if not ok:
            code, msg = self._mt5.last_error()
            logger.error(
                f"mt5.login() failed: code={code} msg='{msg}'\n"
                f"  account={self.login} | server={self.server}\n"
                f"\n  DIAGNOSIS:\n"
                f"  code=10013 → wrong ICM_LOGIN or ICM_PASSWORD\n"
                f"  code=10006 → server name wrong\n"
                f"\n  FIX:\n"
                f"  1. Verify login/password: open MT5 → File → Login to Trade Account\n"
                f"  2. Check server name: try ICMarketsSC-Demo or ICMarketsEU-Demo\n"
                f"  3. Update ICM_LOGIN / ICM_PASSWORD in demo_runner.py")
            self._mt5.shutdown()
            return False

        info = self._mt5.account_info()
        self._connected = True
        tag = 'DEMO' if self.is_demo else '⚠️ LIVE'
        logger.info(
            f"✅ [{tag}] Connected | login={self.login} | "
            f"server={self.server} | balance={info.balance:.2f} {info.currency}")
        return True

    def reconnect(self, max_attempts: int = 5, wait_seconds: int = 30) -> bool:
        self._connected = False
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Reconnect attempt {attempt}/{max_attempts}...")
            try:
                if self._mt5: self._mt5.shutdown()
            except Exception:
                pass
            time.sleep(wait_seconds)
            if self.connect():
                logger.info("✅ Reconnected")
                return True
        logger.error(f"Failed after {max_attempts} attempts")
        return False

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        if not self._connected or not self._mt5: return False
        try:
            info = self._mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    def get_bars(self, symbol: str = None, timeframe: str = 'H4',
                 count: int = 250) -> pd.DataFrame:
        self._require_connected()
        sym = symbol or self.symbol; mt5 = self._mt5
        tf_map = {
            'M1':mt5.TIMEFRAME_M1, 'M5':mt5.TIMEFRAME_M5,
            'M15':mt5.TIMEFRAME_M15, 'M30':mt5.TIMEFRAME_M30,
            'H1':mt5.TIMEFRAME_H1, 'H4':mt5.TIMEFRAME_H4, 'D1':mt5.TIMEFRAME_D1,
        }
        rates = mt5.copy_rates_from_pos(sym, tf_map.get(timeframe, mt5.TIMEFRAME_H4), 0, count)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            logger.error(f"get_bars({sym},{timeframe}): code={code} {msg}")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df['spread'] = df['spread'] / 10.0
        return df

    def get_spread(self, symbol: str = None) -> float:
        self._require_connected()
        info = self._mt5.symbol_info(symbol or self.symbol)
        return float(info.spread) / 10.0 if info else 1.5

    def get_balance(self) -> float:
        self._require_connected()
        info = self._mt5.account_info()
        return float(info.balance) if info else 0.0

    def is_market_open(self, symbol: str = None) -> bool:
        if not self._connected: return False
        info = self._mt5.symbol_info(symbol or self.symbol)
        return bool(info and info.trade_mode == 4)

    def open_order(self, direction: str, lots: float,
                   sl: float, tp: float,
                   symbol: str = None, comment: str = 'v18.7') -> Optional[int]:
        self._require_connected()
        sym = symbol or self.symbol; mt5 = self._mt5
        tick = mt5.symbol_info_tick(sym)
        if not tick: logger.error(f"No tick for {sym}"); return None
        otype = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 'BUY' else tick.bid
        req = {
            'action': mt5.TRADE_ACTION_DEAL, 'symbol': sym,
            'volume': round(lots, 2), 'type': otype, 'price': price,
            'sl': round(sl, 5), 'tp': round(tp, 5),
            'deviation': 20, 'magic': self.magic, 'comment': comment,
            'type_time': mt5.ORDER_TIME_GTC, 'type_filling': mt5.ORDER_FILLING_IOC,
        }
        for attempt in range(3):
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"✅ ORDER {direction} {sym} | lots={lots:.2f} | "
                    f"price={price:.5f} | sl={sl:.5f} | tp={tp:.5f} | "
                    f"ticket={result.order}")
                return int(result.order)
            code = result.retcode if result else 'None'
            logger.warning(f"Order attempt {attempt+1}/3 failed: retcode={code}")
            if attempt < 2: time.sleep(1)
        logger.error("❌ Order failed after 3 attempts")
        return None

    def close_position(self, ticket: int, lots: float,
                       direction: str, symbol: str = None) -> bool:
        self._require_connected()
        sym = symbol or self.symbol; mt5 = self._mt5
        ct  = mt5.ORDER_TYPE_SELL if direction == 'BUY' else mt5.ORDER_TYPE_BUY
        tick= mt5.symbol_info_tick(sym)
        price = tick.bid if direction == 'BUY' else tick.ask
        req = {
            'action': mt5.TRADE_ACTION_DEAL, 'symbol': sym,
            'volume': round(lots, 2), 'type': ct, 'price': price,
            'position': ticket, 'deviation': 20, 'magic': self.magic,
            'comment': 'v18.7_close', 'type_filling': mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        ok = result and result.retcode == mt5.TRADE_RETCODE_DONE
        logger.info(f"{'✅' if ok else '❌'} CLOSE ticket={ticket} price={price:.5f}")
        return ok

    def partial_close(self, ticket, close_lots, direction, symbol=None):
        return self.close_position(ticket, close_lots, direction, symbol)

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        self._require_connected()
        req = {'action': self._mt5.TRADE_ACTION_SLTP,
               'position': ticket, 'sl': round(new_sl, 5)}
        result = self._mt5.order_send(req)
        ok = result and result.retcode == self._mt5.TRADE_RETCODE_DONE
        logger.info(f"{'✅' if ok else '❌'} MODIFY_SL ticket={ticket} sl={new_sl:.5f}")
        return ok

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError(
                "Not connected. Call broker.connect() first.\n"
                "MT5 terminal must be open and logged in.")