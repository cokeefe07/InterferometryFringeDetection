#!/usr/bin/env bash
# One-click launcher for macOS / Linux: creates a local environment on first run.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating Python environment..."
  python3 -m venv .venv || { echo "Could not create the environment (need Python 3.9+ with tkinter)."; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing requirements failed."; exit 1; }
fi
exec .venv/bin/python SpellBook.py
