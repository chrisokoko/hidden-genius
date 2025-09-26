#!/usr/bin/env python3
"""
CLI for fingerprint service.

Usage: python fingerprint_service <input> <output_dir>

Examples:
    python fingerprint_service document output/
    python fingerprint_service transcripts/ fingerprints/
"""

import argparse
import logging
import sys
from pathlib import Path

from .service import process_transcripts


def setup_logging(verbose: bool = False):
    """Setup basic logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate semantic fingerprints from any text content using Claude",
        epilog="""
Examples:
  %(prog)s document output/                   # Process single file
  %(prog)s documents/ fingerprints/           # Process directory
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Input file or directory to fingerprint"
    )

    parser.add_argument(
        "output_dir",
        help="Output directory for fingerprints"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Convert arguments to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Validate input
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)


    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Process based on input type
    try:
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
        else:
            print(f"Error: {input_path} is neither a file nor a directory", file=sys.stderr)
            sys.exit(1)

        result = process_transcripts(input_path, output_dir)

        print(f"\n✅ Processed: {result['processed']}")
        print(f"❌ Failed: {result['failed']}")
        print(f"⏭️ Skipped: {result['skipped']}")

        if result['failed'] > 0:
            sys.exit(1)
        elif result['processed'] == 0:
            print("No files were processed")
            sys.exit(1)
        else:
            print(f"\n✅ Processing complete! Fingerprints saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()