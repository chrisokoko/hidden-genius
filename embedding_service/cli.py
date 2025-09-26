#!/usr/bin/env python3
"""
CLI for embedding service.

Usage: python embedding_service <input> <output_dir>

Examples:
    python embedding_service fingerprint.json output/
    python embedding_service fingerprints/ embeddings/
"""

import argparse
import logging
import sys
from pathlib import Path

from .service import generate_embeddings
from .config import EmbeddingConfig


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
        description="Generate embeddings from semantic fingerprints using OpenAI",
        epilog="""
Examples:
  %(prog)s fingerprint.json output/          # Process single file
  %(prog)s fingerprints/ embeddings/         # Process directory
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Input fingerprint file or directory"
    )

    parser.add_argument(
        "output_dir",
        help="Output directory for embeddings"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "-s", "--strategy",
        default="default",
        help="Embedding strategy to use (default, core_questions, similarity, themes)"
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

    # Check if input is valid type
    if input_path.is_file() and input_path.suffix.lower() != '.json':
        print(f"Error: Input file must be a .json file, got: {input_path.suffix}", file=sys.stderr)
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

        # Validate configuration before processing
        try:
            config = EmbeddingConfig()
            # Override strategy from CLI if provided
            config.active_strategy = args.strategy
        except Exception as config_error:
            print(f"Configuration error: {config_error}", file=sys.stderr)
            print("Please check your .env file and environment variables.", file=sys.stderr)
            sys.exit(1)

        result = generate_embeddings(input_path, output_dir, config)

        print(f"\n✅ Processed: {result['processed']}")
        print(f"❌ Failed: {result['failed']}")
        print(f"⏭️ Skipped: {result['skipped']}")

        if result['failed'] > 0:
            sys.exit(1)
        elif result['processed'] == 0 and result['skipped'] == 0:
            print("No files were processed")
            sys.exit(1)
        else:
            print(f"\n✅ Processing complete! Embeddings saved to: {output_dir}")

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