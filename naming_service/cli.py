"""
Command line interface for the naming service.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import NamingConfig
from .service import NamingService


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Claude-powered names and descriptions for enriched clusters"
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to cluster file or directory of cluster files to process"
    )

    parser.add_argument(
        "output_path",
        type=Path,
        help="Path where named cluster file(s) will be saved (file or directory)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Load configuration
    try:
        config = NamingConfig()
        logging.info("Configuration loaded successfully")

        if args.verbose:
            logging.debug(f"Config: prompts_dir={config.prompts_dir}")
            logging.debug(f"Config: claude_model={config.claude_model}")

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize service
    service = NamingService(config)

    # Determine processing mode
    try:
        if args.input_path.is_file():
            # Single file processing
            logging.info(f"Processing single file: {args.input_path}")
            logging.info(f"Output file: {args.output_path}")
            result = service.process_file(args.input_path, args.output_path)

        elif args.input_path.is_dir():
            # Directory processing
            logging.info(f"Processing directory: {args.input_path}")
            logging.info(f"Output directory: {args.output_path}")
            result = service.process_directory(args.input_path, args.output_path)

        else:
            logging.error(f"Input path does not exist: {args.input_path}")
            sys.exit(1)

        # Print results
        print(f"✅ Processed: {result['processed']}")
        print(f"❌ Failed: {result['failed']}")
        print(f"⏭️ Skipped: {result['skipped']}")

        if result['failed'] == 0:
            print(f"\\n✅ Naming complete! Results saved to: {args.output_path}")
        else:
            print(f"\\n❌ Naming failed. Check logs for details.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Naming service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()