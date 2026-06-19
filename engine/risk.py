from typing import List
from utils.config import RISK


def kelly_fraction(trade_pnls: List[float], kelly_frac: float = RISK.kelly_fraction,
                    max_pos: float = RISK.max_position_pct,
                    min_pos: float = RISK.min_position_pct,
                    min_trades: int = RISK.kelly_min_trades) -> float:
    if len(trade_pnls) < min_trades:
        return max_pos

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]

    if not wins or not losses:
        return max_pos

    win_rate = len(wins) / len(trade_pnls)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))

    if avg_loss == 0:
        return max_pos

    b = avg_win / avg_loss
    f = (win_rate * b - (1 - win_rate)) / b

    f = max(0.0, min(1.0, f)) * kelly_frac
    return max(min_pos, min(max_pos, f))


def position_size(trade_pnls: List[float], use_kelly: bool = RISK.use_kelly_sizing,
                   kelly_frac: float = RISK.kelly_fraction,
                   max_pos: float = RISK.max_position_pct,
                   min_pos: float = RISK.min_position_pct,
                   min_trades: int = RISK.kelly_min_trades) -> float:
    if not use_kelly:
        return max_pos
    return kelly_fraction(trade_pnls, kelly_frac=kelly_frac, max_pos=max_pos,
                          min_pos=min_pos, min_trades=min_trades)


def stop_loss_hit(entry_px: float, current_px: float,
                   stop_loss_pct: float = RISK.stop_loss_pct) -> bool:
    if entry_px <= 0:
        return False
    return current_px <= entry_px * (1 - stop_loss_pct)
