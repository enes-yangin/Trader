import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from utils.config import DATA
from utils.logger import get_logger

log = get_logger("paper_trader")

LOGS_DIR = "logs"
TRADES_FILE = os.path.join(LOGS_DIR, "paper_trades.json")
ACCURACY_LOG = os.path.join(LOGS_DIR, "live_accuracy.log")

def _ensure_logs_dir():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

def load_trades():
    _ensure_logs_dir()
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading paper trades file: {e}")
        return []

def save_trades(trades):
    _ensure_logs_dir()
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Error saving paper trades file: {e}")

def log_prediction(symbol: str, entry_price: float, consensus: str, confidence: float,
                   buy_threshold: float, sell_threshold: float, horizon_days: int = 5):
    """Logs a new live prediction to paper_trades.json and live_accuracy.log."""
    _ensure_logs_dir()
    trades = load_trades()
    
    now = datetime.utcnow()
    resolution_time = now + timedelta(days=horizon_days)
    
    new_trade = {
        "id": len(trades) + 1,
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "entry_price": float(entry_price),
        "consensus": consensus,
        "confidence": float(confidence),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "horizon_days": int(horizon_days),
        "resolution_time": resolution_time.isoformat(),
        "status": "pending",
        "exit_price": None,
        "actual_return": None,
        "actual_label": None,
        "success": None
    }
    
    trades.append(new_trade)
    save_trades(trades)
    
    # Log to live_accuracy.log
    try:
        with open(ACCURACY_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] NEW PREDICTION | "
                f"Symbol: {symbol} | Entry Price: {entry_price:,.2f} | "
                f"Signal: {consensus} (Conf: {confidence:.1f}%) | "
                f"Horizon: {horizon_days}d | "
                f"Buy Th: {buy_threshold:.4%}, Sell Th: {sell_threshold:.4%}\n"
            )
    except Exception as e:
        log.error(f"Failed to write to accuracy log: {e}")
        
    log.info(f"Logged {symbol} prediction: {consensus} @ {entry_price}")
    return new_trade

def evaluate_past_predictions():
    """Checks all pending predictions, resolves them if their horizon has passed,
    and returns a summary statistics dict.
    """
    _ensure_logs_dir()
    trades = load_trades()
    pending_trades = [t for t in trades if t["status"] == "pending"]
    
    if not pending_trades:
        return _compute_stats(trades)
        
    now = datetime.utcnow()
    resolved_any = False
    
    # Group pending trades by symbol to fetch data efficiently
    by_symbol = {}
    for t in pending_trades:
        res_time = datetime.fromisoformat(t["resolution_time"])
        if now >= res_time:
            by_symbol.setdefault(t["symbol"], []).append(t)
            
    if not by_symbol:
        return _compute_stats(trades)
        
    ex = getattr(ccxt, DATA.exchange_id)({"enableRateLimit": True})
    
    for symbol, sym_trades in by_symbol.items():
        try:
            # Fetch daily candles to resolve daily horizon predictions
            log.info(f"Fetching candles for {symbol} to resolve {len(sym_trades)} trades...")
            raw = ex.fetch_ohlcv(symbol, timeframe="1d", limit=100)
            df = pd.DataFrame(raw, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df.set_index("date", inplace=True)
            
            for t in sym_trades:
                trade_date = pd.to_datetime(datetime.fromisoformat(t["timestamp"])).normalize()
                target_date = trade_date + timedelta(days=t["horizon_days"])
                
                # Check if target date candle is in the dataframe
                if target_date in df.index:
                    exit_price = float(df.loc[target_date, "close"])
                    entry_price = t["entry_price"]
                    actual_return = (exit_price - entry_price) / entry_price
                    
                    # Classify actual label based on the recorded thresholds
                    if actual_return > t["buy_threshold"]:
                        actual_label = "BUY"
                    elif actual_return < t["sell_threshold"]:
                        actual_label = "SELL"
                    else:
                        actual_label = "HOLD"
                        
                    success = (actual_label == t["consensus"])
                    
                    t["exit_price"] = exit_price
                    t["actual_return"] = actual_return
                    t["actual_label"] = actual_label
                    t["success"] = success
                    t["status"] = "resolved"
                    resolved_any = True
                    
                    # Log resolved trade
                    msg = (
                        f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] RESOLVED | "
                        f"ID: {t['id']} | Symbol: {symbol} | "
                        f"Entry: {entry_price:,.2f} | Exit: {exit_price:,.2f} | "
                        f"Return: {actual_return:+.4%} | "
                        f"Predicted: {t['consensus']} | Actual: {actual_label} | "
                        f"Result: {'SUCCESS' if success else 'FAILURE'}\n"
                    )
                    with open(ACCURACY_LOG, "a", encoding="utf-8") as f:
                        f.write(msg)
                    log.info(f"Resolved trade ID {t['id']} ({symbol}): {'SUCCESS' if success else 'FAILURE'}")
                else:
                    # If target date is not in index but target_date is in the past relative to df's latest index,
                    # and more than 7 days have passed since resolution time, mark as expired.
                    latest_date = df.index.max()
                    if target_date < latest_date:
                        t["status"] = "expired"
                        resolved_any = True
                        with open(ACCURACY_LOG, "a", encoding="utf-8") as f:
                            f.write(
                                f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] EXPIRED | "
                                f"ID: {t['id']} | Symbol: {symbol} | "
                                f"Target Date {target_date.strftime('%Y-%m-%d')} not found in history.\n"
                            )
                        log.info(f"Trade ID {t['id']} ({symbol}) expired (target date not found).")
                        
        except Exception as e:
            log.error(f"Error resolving trades for {symbol}: {e}")
            
    if resolved_any:
        save_trades(trades)
        
    return _compute_stats(trades)

def _compute_stats(trades):
    resolved = [t for t in trades if t["status"] == "resolved"]
    total = len(resolved)
    correct = sum(1 for t in resolved if t["success"] is True)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    
    stats = {
        "total_predictions": len(trades),
        "total_resolved": total,
        "correct_resolved": correct,
        "accuracy_pct": accuracy,
        "pending_resolved": len([t for t in trades if t["status"] == "pending"])
    }
    
    log.info(f"Paper trading stats: {correct}/{total} resolved correct ({accuracy:.1f}%)")
    return stats
