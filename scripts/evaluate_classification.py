import argparse
import warnings
from data.indicators import engineer, get_features
from data.labeling import add_labels, CLASS_TARGET_COL, LABEL_BUY, LABEL_SELL
from models.classifier_models import LogisticModel, XGBClassifierModel
from utils.config import FEATURES, MODEL

warnings.filterwarnings("ignore")

CLASSIFIERS = {"logistic": LogisticModel, "xgb_clf": XGBClassifierModel}


def directional_hit_rate(preds, fwd_returns):
    mask = (preds == LABEL_BUY) | (preds == LABEL_SELL)
    n_sig = int(mask.sum())
    if n_sig == 0:
        return 0, float("nan")
    correct = 0
    for p, f in zip(preds[mask], fwd_returns[mask]):
        if p == LABEL_BUY and f > 0:
            correct += 1
        elif p == LABEL_SELL and f < 0:
            correct += 1
    return n_sig, correct / n_sig * 100


def prep(sym, years, seed, threshold):
    from tests.realistic_data import make_realistic_ohlcv
    df = make_realistic_ohlcv(n=years * 365, seed=seed, symbol=sym)
    df = engineer(df, with_news=False, with_micro=True)
    df = add_labels(df, threshold=threshold, atr_normalize=True)
    fwd = df[FEATURES.close_col].pct_change(MODEL.pred_horizon).shift(-MODEL.pred_horizon)
    df["_fwd"] = fwd
    df = df.dropna(subset=[CLASS_TARGET_COL, "_fwd"])
    X = get_features(df, with_micro=True).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    fwd_arr = df["_fwd"].values
    n = len(X)
    i_tr, i_va = int(n * 0.7), int(n * 0.85)
    return (X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va],
            X[i_va:], y[i_va:], fwd_arr[i_va:])


def main():
    ap = argparse.ArgumentParser(description="Classification directional accuracy")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--walkforward", action="store_true",
                    help="evaluate across rolling windows instead of a single split")
    ap.add_argument("--labeling", default="fixed", choices=["fixed", "triple_barrier"],
                    help="labeling scheme for walk-forward mode")
    ap.add_argument("--embargo", type=int, default=0,
                    help="extra purge buffer beyond the label horizon (walk-forward)")
    args = ap.parse_args()

    if args.walkforward:
        _run_walkforward(args)
        return

    print("CLASSIFICATION DIRECTIONAL ACCURACY (realistic synthetic data)")
    print("Directional hit rate = of all BUY/SELL signals, fraction where the")
    print("realized move went the predicted way. ~0.50 is coin-flip; sustained")
    print(">0.55 on out-of-sample data would be a meaningful edge.\n")
    print(f"{'symbol':10s} {'model':14s} {'signals':>8s} {'dir_hit%':>9s}")
    print("-" * 45)

    import zlib
    for sym in args.symbols:
        seed = zlib.crc32(sym.encode()) % 1000
        Xtr, ytr, Xv, yv, Xte, yte, fwdte = prep(sym, args.years, seed, args.threshold)
        for name, cls in CLASSIFIERS.items():
            m = cls()
            m.train(Xtr, ytr, Xv, yv)
            preds = m.predict(Xte)
            n_sig, hit = directional_hit_rate(preds, fwdte)
            print(f"{sym:10s} {name:14s} {n_sig:8d} {hit:9.1f}")


def _run_walkforward(args):
    import zlib
    from tests.realistic_data import make_realistic_ohlcv
    from engine.classification_walkforward import run_classification, format_classification_report
    from utils.types import FeatureSpec

    print("CLASSIFICATION WALK-FORWARD (realistic synthetic data)")
    print("Evaluating across rolling windows exposes whether an apparent edge is")
    print("consistent or just one lucky split. Watch 'windows > 50%' and the")
    print("hit-rate std: a real edge is stable across windows, not occasional.\n")

    spec = FeatureSpec.from_bools(False, True, False)
    for sym in args.symbols:
        seed = zlib.crc32(sym.encode()) % 1000
        df = engineer(make_realistic_ohlcv(n=args.years * 365, seed=seed, symbol=sym),
                      with_news=False, with_micro=True)
        print(f"########## {sym} ##########")
        for name in CLASSIFIERS:
            res = run_classification(df, name, spec=spec, threshold=args.threshold,
                                      labeling=args.labeling, embargo=args.embargo,
                                      train_size=400, test_size=100)
            print(format_classification_report(name, res))
        print()


if __name__ == "__main__":
    main()
