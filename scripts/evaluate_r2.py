import argparse
import warnings
import numpy as np
from data.dataset import build
from data.indicators import engineer
from engine.trainer import split
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel
from utils.config import MODEL

warnings.filterwarnings("ignore")

MODELS = {"linear": LinearModel, "xgboost": XGBModel, "lstm": LSTMModel}


def naive_zero_r2(y):
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1 - float(np.sum(y ** 2)) / ss_tot


def directional_accuracy(preds, y):
    n = min(len(preds), len(y))
    return float(np.mean(np.sign(preds[:n]) == np.sign(y[:n])))


def evaluate_symbol(sym, years, allow_sample):
    if allow_sample:
        import zlib
        from tests.realistic_data import make_realistic_ohlcv
        seed = zlib.crc32(sym.encode()) % 1000
        df = make_realistic_ohlcv(n=years * 365, seed=seed, symbol=sym)
    else:
        df = build(sym, years=years, with_news=False, allow_sample=False)
    df = engineer(df, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    src = df.attrs.get("source", "real")
    print(f"\n=== {sym}  (rows={len(df)}, source={src}) ===")
    print(f"  target std (test): {sp['y_test'].std():.5f}")
    print(f"  naive zero-pred test R2: {naive_zero_r2(sp['y_test']):+.4f}")
    print(f"  {'model':10s} {'train_R2':>10s} {'test_R2':>10s} "
          f"{'test_RMSE':>10s} {'dir_acc':>8s}")

    for name, cls in MODELS.items():
        mdl = cls(epochs=MODEL.lstm_params["epochs"]) if name == "lstm" else cls()
        mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
        tr = mdl.evaluate(sp["X_tr"], sp["y_tr"])
        te = mdl.evaluate(sp["X_test"], sp["y_test"])
        preds = mdl.predict(sp["X_test"])
        da = directional_accuracy(preds, sp["y_test"])
        print(f"  {name:10s} {tr['r2']:+10.4f} {te['r2']:+10.4f} "
              f"{te['rmse']:10.5f} {da:8.3f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate model R2 on real market data")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--allow-sample", action="store_true",
                    help="use realistic synthetic data instead of fetching (offline mode)")
    args = ap.parse_args()

    print("R2 EVALUATION ON MARKET DATA")
    print("Interpreting results:")
    print("  - Out-of-sample R2 near 0 (or slightly negative) is NORMAL and")
    print("    HONEST for 5-day-ahead return prediction; markets are near-efficient.")
    print("  - A large train-vs-test R2 gap indicates overfitting, not skill.")
    print("  - Directional accuracy should sit near 0.50; values like 0.70+")
    print("    on test data usually mean a look-ahead leak, not a working edge.")

    for sym in args.symbols:
        try:
            evaluate_symbol(sym, args.years, args.allow_sample)
        except Exception as e:
            print(f"\n=== {sym}: FAILED ({type(e).__name__}: {e}) ===")
            print("  Re-run with --allow-sample to use synthetic data offline.")


if __name__ == "__main__":
    main()
