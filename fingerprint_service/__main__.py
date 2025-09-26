#!/usr/bin/env python3
"""
Main entry point for fingerprint service.

This allows running the service directly with:
    python3 fingerprint_service <input_path> <output_path>
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingerprint_service.cli import main

if __name__ == "__main__":
    main()