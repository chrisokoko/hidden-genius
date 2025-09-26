"""
Command line interface for the enrichment service.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import EnrichmentConfig
from .service import EnrichmentService


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
        description="Enrich clustering data with essence data from fingerprints"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing cluster assignments"
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where enriched results will be saved"
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
        config = EnrichmentConfig()
        logging.info("Configuration loaded successfully")

        if args.verbose:
            logging.debug(f"Config: fingerprints_dir={config.fingerprints_dir}")
            logging.debug(f"Config: claude_model={config.claude_model}")

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize and run service
    try:
        logging.info(f"Processing clusters from: {args.clusters_dir}")
        logging.info(f"Output directory: {args.output_dir}")

        service = EnrichmentService(config)
        result = service.run(args.clusters_dir, args.output_dir)

        # Print results
        print(f"✅ Processed: {result['processed']}")
        print(f"❌ Failed: {result['failed']}")
        print(f"⏭️ Skipped: {result['skipped']}")

        if result['failed'] == 0:
            print(f"\\n✅ Enrichment complete! Results saved to: {args.output_dir}")
        else:
            print(f"\\n❌ Enrichment failed. Check logs for details.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Enrichment service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()