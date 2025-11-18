#!/usr/bin/env python3
"""Simple test runner to exercise graph package after refactor.

This script ensures `src/` is on sys.path then imports the packaged main and runs
with small parameters (n_arbo=1, n_copy=1) so you can quickly verify the
refactor worked.
"""
import os
import sys

# Ensure src/ is importable when running this script from project root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(REPO_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from graph.improved_main import main

if __name__ == '__main__':
    # Small smoke test
    main(n_arbo=1, n_copy=1, output_path='smoke_output')
