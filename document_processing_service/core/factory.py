"""
Factory class that selects and runs the appropriate processor for each file type.
"""

import logging
from pathlib import Path
from typing import List

from ..processors.audio import AudioProcessor
from ..processors.pdf import PDFProcessor
from ..processors.docx import DOCXProcessor
from .base import BaseProcessor
from .types import ProcessorResult


# Setup logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Factory that selects and runs the appropriate processor for each file type.

    This is the main entry point for the document processing service.
    It automatically detects file types and delegates to the appropriate processor.
    """

    def __init__(self):
        """Initialize the factory with all available processors."""
        # Initialize all processors
        self.processors = [
            AudioProcessor(),
            PDFProcessor(),
            DOCXProcessor(),
        ]

    def process_file(self, file_path: Path, output_dir: Path) -> ProcessorResult:
        """
        Process a single file using the appropriate processor.

        Args:
            file_path: Path to the file to process
            output_dir: Directory where output files should be saved

        Returns:
            ProcessorResult object containing the processing results
        """
        # Convert to Path objects if they aren't already
        file_path = Path(file_path)
        output_dir = Path(output_dir)

        # Find the right processor
        for processor in self.processors:
            if processor.can_process(file_path):
                print(f"Processing {file_path.name}... ", end="", flush=True)
                result = processor.process(file_path, output_dir)

                # Print result
                if result.success:
                    print("done")
                else:
                    print(f"failed ({result.error_message})")

                return result

        # No processor found for this file type
        print(f"Skipping {file_path.name} (unsupported type)")
        return ProcessorResult(
            success=False,
            text="",
            source_file=file_path,
            output_file=None,
            processor_type="unknown",
            processing_time=0,
            error_message="No processor available for this file type"
        )

    def process_directory(self, input_dir: Path, output_dir: Path) -> List[ProcessorResult]:
        """
        Process all files in a directory.

        Args:
            input_dir: Directory containing files to process
            output_dir: Directory where output files should be saved

        Returns:
            List of ProcessorResult objects for each file processed
        """
        # Convert to Path objects if they aren't already
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all files in the directory
        files = [f for f in input_dir.iterdir() if f.is_file()]

        if not files:
            print(f"No files found in {input_dir}")
            return []

        print(f"Found {len(files)} files in {input_dir}")

        # Process each file
        results = []
        for file_path in files:
            result = self.process_file(file_path, output_dir)
            results.append(result)

        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        skipped = sum(1 for r in results if r.error_message == "No processor available for this file type")

        print(f"\nComplete: {successful} processed, {failed} failed")
        if skipped > 0:
            print(f"Skipped {skipped} unsupported file types")

        # Log details of failed files
        failed_files = [r for r in results if not r.success and r.error_message != "No processor available for this file type"]
        if failed_files:
            print("\nFailed files:")
            for result in failed_files:
                print(f"  {result.source_file.name}: {result.error_message}")

        return results

    def get_supported_extensions(self) -> List[str]:
        """
        Get a list of all supported file extensions.

        Returns:
            List of supported file extensions
        """
        extensions = []
        for processor in self.processors:
            if hasattr(processor, 'SUPPORTED_EXTENSIONS'):
                extensions.extend(processor.SUPPORTED_EXTENSIONS)

        # Add other known extensions
        extensions.extend(['.pdf', '.docx', '.doc'])

        return sorted(list(set(extensions)))