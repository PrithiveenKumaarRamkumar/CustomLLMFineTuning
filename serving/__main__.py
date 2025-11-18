#!/usr/bin/env python3
"""
CustomLLM Inference API - Main Module Entry Point

This allows running the serving module directly:
    python -m serving

Equivalent to:
    python serving/start_api.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from serving.start_api import main

if __name__ == "__main__":
    main()