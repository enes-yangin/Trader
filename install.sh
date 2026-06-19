#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON_BIN="python3"
if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
    echo "Python 3.10+ not found. Install it from https://www.python.org/downloads/"
    exit 1
fi

$PYTHON_BIN bootstrap.py "$@"
