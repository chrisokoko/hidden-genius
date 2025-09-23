#!/usr/bin/env python3
"""
Simple CLI for document processing service.

Usage: python cli.py <input> <output_dir>

Examples:
    python cli.py document.pdf output/
    python cli.py documents/ output/
"""

import argparse
import logging
import sys
from pathlib import Path

from .core.factory import DocumentProcessor


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
        description="Convert any document to text",
        epilog="""
Examples:
  %(prog)s document.pdf output/          # Process single file
  %(prog)s documents/ output/            # Process entire directory
  %(prog)s audio.m4a transcripts/        # Transcribe audio file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Input file or directory to process"
    )

    parser.add_argument(
        "output_dir",
        help="Output directory for text files"
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

    # Initialize processor
    processor = DocumentProcessor()

    # Show supported file types if verbose
    if args.verbose:
        extensions = processor.get_supported_extensions()
        logger.info(f"Supported file types: {', '.join(extensions)}")

    # Process based on input type
    try:
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            result = processor.process_file(input_path, output_dir)

            if result.success:
                print(f"\n✅ Success! Text saved to: {result.output_file}")
            else:
                print(f"\n❌ Failed: {result.error_message}")
                sys.exit(1)

        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            results = processor.process_directory(input_path, output_dir)

            if not results:
                print("No files were processed")
                sys.exit(1)

            # Check if any processing was successful
            successful = any(r.success for r in results)
            if not successful:
                print("\n❌ No files were successfully processed")
                sys.exit(1)
            else:
                print(f"\n✅ Processing complete! Text files saved to: {output_dir}")

        else:
            print(f"Error: {input_path} is neither a file nor a directory", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()