#!/usr/bin/env python3
"""Wrapper to run src.cycle with project root added to sys.path.
This avoids ModuleNotFoundError for `src` when running from different CWDs.
"""
import sys
import pathlib
import traceback

root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root))
print(f"Inserted project root into sys.path: {root}")

try:
    # Importing the module will execute its top-level script code
    import src2.cycle  # noqa: F401
except Exception:
    traceback.print_exc()
    sys.exit(1)
