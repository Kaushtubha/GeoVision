"""CLI entrypoint for GeoVision operations."""

import argparse
import sys

from geovision import __version__
from geovision.logger import get_logger

logger = get_logger("geovision.cli")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="geovision",
        description="GeoVision: Multimodal Satellite-Intelligence Platform CLI",
    )
    parser.add_argument("--version", "-v", action="version", version=f"GeoVision v{__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Info subcommand
    subparsers.add_parser("info", help="Display GeoVision system and environment information")

    # Download subcommand
    dl_parser = subparsers.add_parser("download", help="Download datasets")
    dl_parser.add_argument("--dataset", default="all", choices=["all", "eurosat", "nwpu_vhr10", "landcover_ai", "levir_cd"])
    dl_parser.add_argument("--smoke", action="store_true", help="Generate lightweight smoke dataset")

    args = parser.parse_args()

    if args.command == "info":
        logger.info(f"GeoVision Platform v{__version__}")
        logger.info(f"Python Runtime: {sys.version.split()[0]}")
    elif args.command == "download":
        from geovision.data.downloaders import download_dataset
        download_dataset(args.dataset, smoke=args.smoke)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
