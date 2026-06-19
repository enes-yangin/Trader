import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from portfolio.position import Trade, match_fifo
from utils.config import DATA
from utils.exceptions import PortfolioError
from utils.logger import get_logger

log = get_logger("ledger")

DEFAULT_PATH = Path(DATA.data_dir) / "portfolio" / "ledger.json"


def _trade_to_dict(t: Trade) -> Dict[str, Any]:
    d = asdict(t)
    d["timestamp"] = t.timestamp.isoformat()
    return d


def _trade_from_dict(d: Dict[str, Any]) -> Trade:
    d = dict(d)
    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
    return Trade(**d)


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.trades: List[Trade] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.trades = []
            self._loaded = True
            return
        raw = json.loads(self.path.read_text())
        self.trades = [_trade_from_dict(d) for d in raw]
        self._loaded = True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [_trade_to_dict(t) for t in self.trades]
        self.path.write_text(json.dumps(data, indent=2))

    def add_trade(self, trade: Trade, validate: bool = True) -> None:
        self._ensure_loaded()
        if validate:
            match_fifo(self.trades + [trade])
        self.trades.append(trade)
        self.save()
        log.info(f"added {trade.side} {trade.qty} {trade.symbol} @ {trade.price}")

    def remove_trade(self, index: int) -> Trade:
        self._ensure_loaded()
        if index < 0 or index >= len(self.trades):
            raise PortfolioError(f"No trade at index {index} (ledger has {len(self.trades)})")
        t = self.trades.pop(index)
        self.save()
        log.info(f"removed {t.side} {t.qty} {t.symbol} @ {t.price}")
        return t

    def list_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        self._ensure_loaded()
        if symbol:
            return [t for t in self.trades if t.symbol == symbol]
        return list(self.trades)

    def symbols(self) -> List[str]:
        self._ensure_loaded()
        return sorted({t.symbol for t in self.trades})

    def clear(self) -> None:
        self.trades = []
        self._loaded = True
        self.save()
