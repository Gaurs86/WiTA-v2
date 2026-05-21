#!/usr/bin/env python
"""scripts/train.py — Thin wrapper; delegates to main.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import main
if __name__ == "__main__":
    main()
