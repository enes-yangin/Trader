import argparse
import warnings
import zlib
from data.indicators import engineer
from engine.feature_ablation import run_ablation, summarize_ablation

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Feature group ablation study")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--model", default="xgb_clf", choices=["logistic", "xgb_clf"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("FEATURE ABLATION (realistic synthetic data)")
    print("Adding a feature family is only justified if it lifts mean hit rate")
    print("without inflating its variance. On near-random data, extra families")
    print("usually add noise -- which is the honest finding this surfaces.\n")

    from tests.realistic_data import make_realistic_ohlcv
    for sym in args.symbols:
        seed = zlib.crc32(sym.encode()) % 1000
        df = engineer(make_realistic_ohlcv(n=args.years * 365, seed=seed, symbol=sym),
                      with_news=False, with_micro=True)
        print(f"########## {sym} ##########")
        table = run_ablation(df, model_name=args.model, threshold=args.threshold,
                              train_size=400, test_size=100)
        print(summarize_ablation(table))
        print()


if __name__ == "__main__":
    main()
