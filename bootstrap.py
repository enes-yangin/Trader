import argparse
import os
import subprocess
import sys
from pathlib import Path

MIN_PY = (3, 10)
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def check_base_python() -> None:
    if sys.version_info < MIN_PY:
        cur = f"{sys.version_info.major}.{sys.version_info.minor}"
        req = f"{MIN_PY[0]}.{MIN_PY[1]}"
        sys.exit(
            f"Python {req}+ is required, found {cur}.\n"
            f"Install a newer Python from https://www.python.org/downloads/ "
            f"and re-run this script with it."
        )


def create_venv() -> None:
    if venv_python().exists():
        print(f"[bootstrap] venv already exists at {VENV_DIR}")
        return
    print(f"[bootstrap] creating venv at {VENV_DIR} ...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)


def install_requirements(force: bool = False) -> None:
    py = str(venv_python())
    marker = VENV_DIR / ".deps_installed"
    if marker.exists() and not force:
        print("[bootstrap] dependencies already installed (use --force-reinstall to redo)")
        return
    print("[bootstrap] upgrading pip ...")
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("[bootstrap] installing requirements.txt (this can take several minutes) ...")
    cmd = [py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    if force:
        cmd.append("--force-reinstall")
    subprocess.run(cmd, check=True)
    marker.write_text("ok")


def run_diagnostics() -> bool:
    py = str(venv_python())
    print("[bootstrap] running environment diagnostics ...")
    result = subprocess.run([py, str(ROOT / "runtime_check.py")])
    return result.returncode == 0


def launch_app(extra_args: list) -> None:
    py = str(venv_python())
    print("[bootstrap] launching AI Trader ...")
    subprocess.run([py, str(ROOT / "main.py"), *extra_args], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader bootstrap")
    parser.add_argument("--check-only", action="store_true",
                        help="run environment checks and exit, do not launch the app")
    parser.add_argument("--skip-install", action="store_true",
                        help="skip dependency installation step")
    parser.add_argument("--force-reinstall", action="store_true",
                        help="force reinstall all dependencies")
    args, extra = parser.parse_known_args()

    check_base_python()
    create_venv()
    if not args.skip_install:
        install_requirements(force=args.force_reinstall)

    ok = run_diagnostics()
    if args.check_only:
        sys.exit(0 if ok else 1)
    if not ok:
        print("[bootstrap] environment check reported issues. Launching anyway ...")
    launch_app(extra)


if __name__ == "__main__":
    main()
