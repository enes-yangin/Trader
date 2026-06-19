import importlib
import socket
import sys
from typing import List, Tuple

MIN_PY = (3, 10)
REQUIRED_MODULES = [
    "ccxt", "pandas", "numpy", "ta",
    "sklearn", "xgboost", "torch", "matplotlib",
]
OPTIONAL_MODULES = [
    "transformers", "vaderSentiment", "feedparser", "optuna", "pyarrow",
]


def check_python_version() -> Tuple[bool, str]:
    ok = sys.version_info >= MIN_PY
    cur = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    req = f"{MIN_PY[0]}.{MIN_PY[1]}+"
    return ok, f"Python {cur} (required {req})"


def check_tkinter() -> Tuple[bool, str]:
    try:
        importlib.import_module("tkinter")
        return True, "tkinter available"
    except ImportError:
        return False, "tkinter NOT available (install python3-tk via your OS package manager)"


def check_module(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return True, f"{name} {ver}"
    except ImportError:
        return False, f"{name} not installed"


def check_torch_device() -> Tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"torch CUDA available ({torch.cuda.get_device_name(0)})"
        return True, "torch CPU only (no CUDA device found, LSTM training will be slower)"
    except ImportError:
        return False, "torch not installed"


def check_network(host: str = "api.binance.com", port: int = 443, timeout: float = 3.0) -> Tuple[bool, str]:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True, f"network reachable ({host})"
    except OSError as e:
        return False, (
            f"cannot reach {host} ({type(e).__name__}). "
            f"Live crypto data unavailable -- app will fall back to sample data."
        )


def run_checks() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    ok, msg = check_python_version()
    results.append(("Python version", ok, msg))

    ok, msg = check_tkinter()
    results.append(("Tkinter (GUI)", ok, msg))

    for name in REQUIRED_MODULES:
        ok, msg = check_module(name)
        results.append((f"Required: {name}", ok, msg))

    for name in OPTIONAL_MODULES:
        ok, msg = check_module(name)
        results.append((f"Optional: {name}", ok, msg))

    ok, msg = check_torch_device()
    results.append(("Torch device", ok, msg))

    ok, msg = check_network()
    results.append(("Network (Binance)", ok, msg))

    return results


def print_report(results: List[Tuple[str, bool, str]]) -> bool:
    print("=" * 60)
    print("  AI Trader - Environment Check")
    print("=" * 60)
    critical_ok = True
    for label, ok, msg in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label:24s} {msg}")
        if not ok and (label.startswith("Required") or label in
                       ("Python version", "Tkinter (GUI)")):
            critical_ok = False
    print("=" * 60)
    if critical_ok:
        print("  All critical checks passed.")
    else:
        print("  Some critical checks FAILED. See FAIL lines above.")
    print("=" * 60)
    return critical_ok


if __name__ == "__main__":
    ok = print_report(run_checks())
    sys.exit(0 if ok else 1)
