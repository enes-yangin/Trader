from typing import List, Optional, Tuple
import numpy as np
from utils.config import RISK


# ------------------------------------------------------------------
# Kelly / Position sizing (mevcut)
# ------------------------------------------------------------------

def kelly_fraction(trade_pnls: List[float], kelly_frac: float = RISK.kelly_fraction,
                    max_pos: float = RISK.max_position_pct,
                    min_pos: float = RISK.min_position_pct,
                    min_trades: int = RISK.kelly_min_trades) -> float:
    if len(trade_pnls) < min_trades:
        return max_pos

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    total_valid = len(wins) + len(losses)
    if total_valid < min_trades:
        return max_pos

    if not wins or not losses:
        return max_pos

    win_rate = len(wins) / total_valid
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


# ------------------------------------------------------------------
# Kaldıraç / Likidasyon Riski (Feature 4)
# ------------------------------------------------------------------

# Varsayılan kaldıraç parametreleri — RiskConfig'ten gelir;
# RiskConfig genişletildikten sonra RISK.xxx olarak okunur.
DEFAULT_MAINTENANCE_MARGIN = 0.005         # %0.5 (BTC için tipik Binance değeri)
DEFAULT_MAX_LEVERAGE = 10.0                # Maks kaldıraç
DEFAULT_FUNDING_RATE_ANNUAL = 0.10         # Yıllık %10 funding (varsayılan)
DEFAULT_LIQ_BUFFER = 0.002                 # Likidasyon fiyatına buffer


def liquidation_price(
    entry_px: float,
    position_size: float,
    leverage: float,
    side: str = "long",
    maintenance_margin_pct: float = DEFAULT_MAINTENANCE_MARGIN,
) -> float:
    """Standard isolated-margin liquidation price (Binance formula).

    Long:  entry * (1 − 1/leverage + maintenance_margin_pct)
    Short: entry * (1 + 1/leverage − maintenance_margin_pct)

    Parameters
    ----------
    entry_px : float
        Position entry price.
    position_size : float
        Position size in USDT (placeholder — does not affect the price ratio).
    leverage : float
        Leverage used (1x = spot-like, >1x = leveraged).
    side : str
        "long" or "short".
    maintenance_margin_pct : float
        Maintenance margin ratio (e.g. 0.005 = 0.5%).

    Returns
    -------
        Bakım marjini yüzdesi (örn. 0.005 = %0.5).

    Returns
    -------
    float
        Likidasyon fiyatı.
    """
    if entry_px <= 0 or leverage <= 0:
        return 0.0

    if side == "long":
        # Standard long isolated margin liquidation: entry * (1 - 1/leverage + MMR)
        liq = entry_px * (1 - 1 / leverage + maintenance_margin_pct)
    elif side == "short":
        # Standard short isolated margin liquidation: entry * (1 + 1/leverage - MMR)
        liq = entry_px * (1 + 1 / leverage - maintenance_margin_pct)
    else:
        return 0.0

    return max(0.0, liq)


def vol_adjusted_max_leverage(
    volatility: float,
    max_lev: float = DEFAULT_MAX_LEVERAGE,
    vol_20d_pct: Optional[float] = None,
) -> float:
    """Volatiliteye bağlı maksimum kaldıraç limiti.

    Yüksek volatilite → daha düşük kaldıraç (likidasyon riskini azaltır).
    Düşük volatilite → daha yüksek kaldıraç (sermaye verimliliği).

    Parameters
    ----------
    volatility : float
        Günlük volatilite (standart sapma).
    max_lev : float
        Mutlak maksimum kaldıraç.
    vol_20d_pct : float veya None
        20 günlük volatilite yüzdesi. None ise 3 * volatility kullanılır.

    Returns
    -------
    float
        Volatilite-ayarlı maks kaldıraç (1.0 − max_lev aralığında).
    """
    if volatility <= 0:
        return max_lev

    annualized_vol = volatility * np.sqrt(365)
    # %80 yıllık vol'da kaldıraç 2x'e düşer, %20'de max'a yaklaşır
    target_vol = 0.40  # %40 yıllık vol = 5x kaldıraç
    adjusted = max_lev * (target_vol / max(annualized_vol, 0.01))
    return max(1.0, min(max_lev, adjusted))


def funding_cost(
    position_value: float,
    funding_rate_annual: float = DEFAULT_FUNDING_RATE_ANNUAL,
    holding_days: float = 1.0,
) -> float:
    """Kaldıraçlı pozisyon için günlük fonlama maliyeti.

    Perpetual futures piyasasında long/short arası dengeyi sağlayan
    oran üzerinden hesaplanır. Spot piyasada 0'dır.

    Parameters
    ----------
    position_value : float
        Pozisyonun USDT cinsinden değeri.
    funding_rate_annual : float
        Yıllık funding oranı (örn. 0.10 = %10).
    holding_days : float
        Beklenen elde tutma süresi (gün).

    Returns
    -------
    float
        USDT cinsinden beklenen fonlama maliyeti.
    """
    daily_rate = funding_rate_annual / 365
    return position_value * daily_rate * holding_days


def check_liquidation(
    entry_px: float,
    current_px: float,
    leverage: float,
    side: str = "long",
    maintenance_margin_pct: float = DEFAULT_MAINTENANCE_MARGIN,
) -> Tuple[bool, float]:
    """Pozisyonun likide olup olmadığını kontrol et.

    Returns
    -------
    Tuple[bool, float]
        (likidasyon_oldu_mu, likidasyon_fiyatı)
    """
    liq_px = liquidation_price(
        entry_px, position_size=1000.0,  # Sadece liq fiyatı için placeholder
        leverage=leverage, side=side,
        maintenance_margin_pct=maintenance_margin_pct,
    )

    if liq_px <= 0:
        return False, liq_px

    if side == "long":
        return current_px <= liq_px, liq_px
    else:
        return current_px >= liq_px, liq_px


def liquidation_buffer_price(
    entry_px: float,
    leverage: float,
    side: str = "long",
    buffer_pct: float = DEFAULT_LIQ_BUFFER,
) -> float:
    """Likidasyon fiyatının buffer kadar uzağında güvenli stop seviyesi.

    Likidasyon olmadan önce pozisyonu kapatmak için kullanılır —
    "Liquidation önleme"nin gerçek mekanizması.
    """
    liq = liquidation_price(entry_px, position_size=1000.0, leverage=leverage, side=side)
    if liq <= 0:
        return 0.0

    if side == "long":
        return liq * (1 + buffer_pct)
    else:
        return liq * (1 - buffer_pct)
