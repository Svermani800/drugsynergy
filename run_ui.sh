#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -x "$VENV_DIR/bin/python" ]; then VENV_DIR="$PROJECT_DIR/../../.venv"; fi
PYTHON_BIN="$VENV_DIR/bin/python"
OPENMP_DIR=$(find "$VENV_DIR/lib" -type d -path '*/site-packages/sklearn/.dylibs' -print -quit)
export DYLD_LIBRARY_PATH="$OPENMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m app.server
