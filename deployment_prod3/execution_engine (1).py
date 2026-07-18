"""
execution_engine.py  —  EUR/USD H4  v18.14
"""

import MetaTrader5 as mt5
import logging

logger = logging.getLogger(__name__)


def execute_trade(signal, lot=0.10, sl=None, tp=None,
                  symbol="EURUSD", magic=1814):
    """
    Places a market order via MT5.
    Returns (success: bool, result_dict)
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("Cannot get tick for %s", symbol)
        return False, {}

    if signal == "BUY":
        price      = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price      = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : lot,
        "type"        : order_type,
        "price"       : price,
        "deviation"   : 10,
        "magic"       : magic,
        "comment"     : "v18.14 AI",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if sl:
        request["sl"] = sl
    if tp:
        request["tp"] = tp

    result = mt5.order_send(request)
    success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    logger.info("Trade %s | lot=%.2f | price=%.5f | retcode=%s",
                signal, lot, price,
                result.retcode if result else "None")

    return success, result._asdict() if result else {}


def close_all_positions(symbol="EURUSD", magic=1814):
    """Close all open positions for the magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return
    for pos in positions:
        if pos.magic != magic:
            continue
        tick = mt5.symbol_info_tick(symbol)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask
        req = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : symbol,
            "volume"      : pos.volume,
            "type"        : close_type,
            "position"    : pos.ticket,
            "price"       : price,
            "deviation"   : 10,
            "magic"       : magic,
            "comment"     : "close v18.14",
            "type_time"   : mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)
