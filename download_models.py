"""
download_models.py
──────────────────
Download ONNX model assets for VisoMaster.

Usage:
    python download_models.py                  # dev mode (default)
    python download_models.py --mode dev       # same as above
    python download_models.py --mode full      # all available models

Modes:
    dev   Downloads only the models that are enabled by default — those
          without "dev": False in models_data.py. These are the minimum
          set needed to run VisoMaster out of the box.

    full  Downloads every model defined in models_data.py, including
          optional ones (SimSwap, GhostFace, CSCS, frame enhancers,
          LivePortrait ONNX, FaceParser, CLIPSeg, etc.).
"""

import argparse

from app.helpers.downloader import download_file
from app.processors.models_data import models_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download VisoMaster model assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="dev",
        help="dev = default models only (default); full = all models",
    )
    args = parser.parse_args()

    if args.mode == "full":
        target = models_list
        print(f"[download_models] Full mode — {len(target)} models to check.")
    else:
        target = [m for m in models_list if m.get("dev", True)]
        print(f"[download_models] Dev mode — {len(target)} default models to check.")

    for model_data in target:
        download_file(
            model_data["model_name"],
            model_data["local_path"],
            model_data["hash"],
            model_data["url"],
        )

    print("[download_models] All done.")


if __name__ == "__main__":
    main()
