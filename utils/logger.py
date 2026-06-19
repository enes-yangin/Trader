import logging
import os
import sys

LOG_DIR = "logs"
LOG_FILE = "ai_trader.log"
LOG_LEVEL = os.environ.get("AI_TRADER_LOG_LEVEL", "INFO")

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger("ai_trader")
    root.setLevel(LOG_LEVEL)
    root.propagate = False

    if root.handlers:
        _configured = True
        return

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(LOG_LEVEL)
    root.addHandler(sh)

    try:
        fh = logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE))
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
    except OSError:
        pass

    _configured = True


def get_logger(name):
    _configure()
    return logging.getLogger(f"ai_trader.{name}")
