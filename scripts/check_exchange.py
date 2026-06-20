"""Hangi borsalara bu makineden erişilebildiğini test eder.

Kullanım:
    .venv/Scripts/python.exe scripts/check_exchange.py

Çıktı: erişilebilen borsalar [OK], engelli/erişilemeyenler [FAIL].
Çalışan bir borsa bulursan, uygulamayı şu env ile başlat:
    Windows (PowerShell):  $env:AI_TRADER_EXCHANGE="kraken"; python main.py
    veya kalıcı: sistem ortam değişkeni olarak AI_TRADER_EXCHANGE=kraken
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt

SYMBOL = "BTC/USDT"
# Yaygın, USDT çiftli, herkese açık (API key gerektirmeyen) borsalar
CANDIDATES = ["binance", "kraken", "kucoin", "okx", "bybit", "coinbase",
              "gateio", "mexc", "bitget", "htx"]


def test(exchange_id: str) -> None:
    t0 = time.perf_counter()
    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        sym = SYMBOL
        # Bazı borsalarda BTC/USDT yerine BTC/USD olabilir
        try:
            ex.load_markets()
            if sym not in ex.markets and "BTC/USD" in ex.markets:
                sym = "BTC/USD"
        except Exception:
            pass
        bars = ex.fetch_ohlcv(sym, timeframe="1d", limit=5)
        dt = time.perf_counter() - t0
        print(f"  [OK]   {exchange_id:10s} {len(bars)} bar ({sym}) — {dt:.1f}s")
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  [FAIL] {exchange_id:10s} {type(e).__name__}: {str(e)[:90]} — {dt:.1f}s")


if __name__ == "__main__":
    print(f"Borsa erişim testi (sembol ~ {SYMBOL}):\n")
    for ex_id in CANDIDATES:
        if hasattr(ccxt, ex_id):
            test(ex_id)
    print("\nÇalışan bir borsa için: AI_TRADER_EXCHANGE=<borsa_adi> ayarla.")
