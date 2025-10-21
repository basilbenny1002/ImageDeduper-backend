# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Basil Benny
# Repository: github.com/basilbenny1002/Image-Selecter
# License notice: Permission is granted to use, copy, modify, and distribute this software
# for any purpose with or without fee, provided that the above notice appears in all copies.

"""
CLI utility to invoke the image selection service directly.
This refactor keeps backward compatibility while delegating heavy lifting
to app.services.image_selector.ImageSelectorService.
"""

import argparse
import sys
from pathlib import Path

from app.services.image_selector import ImageSelectorService



#Setting an environment variable to turn off warnings
def main():
    parser = argparse.ArgumentParser(description="Run image selection locally")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for best images")
    parser.add_argument("--similarity", type=float, default=0.87, help="Similarity threshold (0-1)")
    # Aesthetics is always enabled; no flag to disable
    # No user id required for CLI; we scope to an internal default
    args = parser.parse_args()

    # Safety prompt
    print("WARNING: This operation will modify the input directory.")
    print("- Files may be moved or deleted during deduplication.")
    print("- Make sure you have a backup of your input images before proceeding.\n")
    resp = input("Proceed with processing? [y/N]: ").strip().lower()
    if resp not in ("y", "yes"):
        print("Aborted.")
        sys.exit(1)

    service = ImageSelectorService()
    result = service.choose_best(
        user_id="cli",
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        similarity=args.similarity,
        use_aesthetics=True,
    )
    print(f"Kept: {len(result.kept)} | Removed: {len(result.removed)}")


if __name__ == "__main__":
    main()


    
            



