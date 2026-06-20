"""Her sembol için gerçek-veri yükleme yolunu test eder ve TAM hatayı basar.

Train & Predict "some errors" verince bunu çalıştır:
    .venv\\Scripts\\python.exe scripts\\diagnose_train.py

Çıktıdaki [FAIL] satırlarını paylaş — hangi sembolün neden patladığını gösterir.
"""
import warnings, traceback, sys, os, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DATA
from utils.types import FeatureSpec
from engine.trainer import load_data

print(f"Borsa (exchange_id): {DATA.exchange_id}")
print(f"Semboller: {list(DATA.crypto_symbols)}\n")

# Minimal spec: veri çekimini özelliklerden ayırmak için sadece teknik+micro
spec = FeatureSpec(news=False, micro=True, cross_asset=False)

ok, fail = 0, 0
for sym in DATA.crypto_symbols:
    t0 = time.perf_counter()
    try:
        df = load_data(sym, spec=spec, allow_sample=False)
        sample = bool(df.attrs.get("sample", False))
        tag = " (SAMPLE fallback)" if sample else ""
        print(f"  [OK]   {sym:10s} {len(df):4d} satır{tag} — {time.perf_counter()-t0:.1f}s")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] {sym:10s} {type(e).__name__}: {str(e)[:140]} — {time.perf_counter()-t0:.1f}s")
        fail += 1

print(f"\nSonuç: {ok} OK, {fail} FAIL")
if fail:
    print("\n[FAIL] satırlarındaki sebepleri paylaş. 'market symbol' hatası = o sembol bu borsada yok.")
    print("Tümü FAIL ise: borsa erişimi sorunu — scripts/check_exchange.py ile erişilebilir borsa bul.")
