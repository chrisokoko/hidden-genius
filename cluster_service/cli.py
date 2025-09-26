#!/usr/bin/env python3
"""
CLI for clustering service.

Usage: python cluster_service <embeddings_dir> <output_dir>

Examples:
    python cluster_service embeddings/ clusters/
    python -m cluster_service embeddings/ output/
"""

import argparse
import logging
import sys
from pathlib import Path

from .service import ClusteringService
from .config import ClusteringConfig


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
        description="Perform intelligent hierarchical clustering on embeddings",
        epilog="""
Examples:
  %(prog)s embeddings/ output/                  # Process directory
  %(prog)s embeddings/ clusters/ --verbose      # With verbose logging
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "embeddings_dir",
        help="Directory containing embedding JSON files"
    )

    parser.add_argument(
        "output_dir",
        help="Output directory for clustering results"
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
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    # Validate input
    if not embeddings_dir.exists():
        print(f"Error: {embeddings_dir} not found", file=sys.stderr)
        sys.exit(1)

    if not embeddings_dir.is_dir():
        print(f"Error: {embeddings_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate configuration before processing
    try:
        config = ClusteringConfig()
        logger.info(f"Configuration loaded successfully")
        if args.verbose:
            logger.debug(f"Config: min_clusters={config.min_clusters}, target_count={config.target_count}")
    except Exception as config_error:
        print(f"Configuration error: {config_error}", file=sys.stderr)
        print("Please check your .env file and configuration.", file=sys.stderr)
        sys.exit(1)

    # Process clustering
    try:
        logger.info(f"Processing embeddings from: {embeddings_dir}")
        logger.info(f"Output directory: {output_dir}")

        service = ClusteringService()
        result = service.run(embeddings_dir, output_dir)

        print(f"\n✅ Processed: {result['processed']}")
        print(f"❌ Failed: {result['failed']}")
        print(f"⏭️ Skipped: {result['skipped']}")

        if result['failed'] > 0:
            sys.exit(1)
        elif result['processed'] == 0 and result['skipped'] == 0:
            print("No embeddings were processed")
            sys.exit(1)
        else:
            print(f"\n✅ Clustering complete! Results saved to: {output_dir}")

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