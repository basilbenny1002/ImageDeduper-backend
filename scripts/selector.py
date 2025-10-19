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
from pathlib import Path

from app.services.image_selector import ImageSelectorService



#Setting an environment variable to turn off warnings
def main():
    parser = argparse.ArgumentParser(description="Run image selection locally")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for best images")
    parser.add_argument("--similarity", type=float, default=0.87, help="Similarity threshold (0-1)")
    parser.add_argument("--no-aesthetics", action="store_true", help="Skip aesthetics scoring")
    args = parser.parse_args()

    service = ImageSelectorService()
    result = service.choose_best(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        similarity=args.similarity,
        use_aesthetics=not args.no_aesthetics,
    )
    print(f"Kept: {len(result.kept)} | Removed: {len(result.removed)}")


if __name__ == "__main__":
    main()


    
            



