#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to clean up installation directories.

This script removes old .egg-info directories to avoid conflicts
between the new and old package structure.
"""
import shutil
import os
from pathlib import Path

def main():
    # Get the root directory
    root_dir = Path(__file__).parent.parent

    # Clean up old egg-info directories
    egg_info_dirs = list(root_dir.glob("*.egg-info"))
    if egg_info_dirs:
        print(f"Found {len(egg_info_dirs)} .egg-info directories:")
        for egg_dir in egg_info_dirs:
            print(f"  - {egg_dir}")
            try:
                shutil.rmtree(egg_dir)
                print(f"    ✓ Removed")
            except Exception as e:
                print(f"    ✗ Failed to remove: {e}")

    # Create an empty __init__.py in any empty directories
    for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, "uavarprior")):
        if not filenames and "__init__.py" not in filenames:
            init_path = os.path.join(dirpath, "__init__.py")
            try:
                with open(init_path, "w") as f:
                    pass
                print(f"Created empty __init__.py in {dirpath}")
            except Exception as e:
                print(f"Failed to create __init__.py in {dirpath}: {e}")

if __name__ == "__main__":
    main()
