"""
Factory class that selects and runs the appropriate processor for each file type.
"""

import json
import logging
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ..processors.audio import AudioProcessor
from ..processors.pdf import PDFProcessor
from ..processors.docx import DOCXProcessor
from ..processors.txt import TxtProcessor
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
            TxtProcessor(),
        ]

    def _get_file_creation_date(self, file_path: Path) -> str:
        """
        Get the actual creation date of a file (not modification date).

        Args:
            file_path: Path to the file

        Returns:
            ISO formatted creation date string
        """
        try:
            # Get file stats
            stat = file_path.stat()

            # On Windows, st_ctime is creation time
            # On Unix/Linux/macOS, st_birthtime is creation time (if available)
            if platform.system() == 'Windows':
                creation_time = stat.st_ctime
            else:
                # Try to get birth time (creation time) on Unix systems
                try:
                    creation_time = stat.st_birthtime
                except AttributeError:
                    # Fall back to st_ctime if birthtime not available
                    creation_time = stat.st_ctime

            # Convert to datetime and format as ISO string
            creation_date = datetime.fromtimestamp(creation_time)
            return creation_date.isoformat()

        except Exception:
            # Fallback to current time if we can't get creation date
            return datetime.now().isoformat()

    def _get_file_type(self, file_path: Path) -> str:
        """
        Determine file type from extension.

        Args:
            file_path: Path to the file

        Returns:
            File type string
        """
        suffix = file_path.suffix.lower()

        if suffix in ['.m4a', '.mp3', '.wav', '.ogg', '.flac']:
            return 'audio'
        elif suffix == '.pdf':
            return 'pdf'
        elif suffix in ['.docx', '.doc']:
            return 'docx'
        elif suffix == '.txt':
            return 'txt'
        else:
            return 'unknown'

    def _create_metadata_json(self, result: ProcessorResult) -> Dict[str, Any]:
        """
        Create metadata-only JSON structure.

        Args:
            result: ProcessorResult from a processor

        Returns:
            Dictionary with metadata only (no extracted text)
        """
        # Get file stats
        stat = result.source_file.stat()

        return {
            "file_name": result.source_file.name,
            "file_path": str(result.source_file.absolute()),
            "file_type": self._get_file_type(result.source_file),
            "creation_date": self._get_file_creation_date(result.source_file),
            "file_size": stat.st_size,
            "processing_date": datetime.now().isoformat(),
            "processor_type": result.processor_type,
            "processing_time": result.processing_time,
            "success": result.success
        }

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

        # Create output file paths with subdirectories
        markdown_output_file = output_dir / "markdowns" / f"{file_path.stem}.md"
        json_output_file = output_dir / "jsons" / f"{file_path.stem}.json"

        # Check if output already exists
        if markdown_output_file.exists() and json_output_file.exists():
            print(f"Skipping {file_path.name} (already exists)")
            return ProcessorResult(
                success=True,
                text="",
                source_file=file_path,
                processor_type="cached",
                processing_time=0,
                error_message="File already exists"
            )

        # Find the right processor
        for processor in self.processors:
            if processor.can_process(file_path):
                print(f"Processing {file_path.name}... ", end="", flush=True)

                # Let processor extract text
                result = processor.process(file_path)

                if result.success:
                    # Create output directories
                    markdown_output_file.parent.mkdir(parents=True, exist_ok=True)
                    json_output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Save markdown file with extracted text
                    with open(markdown_output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text)

                    # Create metadata-only JSON
                    metadata_json = self._create_metadata_json(result)

                    # Save JSON file with metadata only
                    with open(json_output_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata_json, f, indent=2)


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
        extensions.extend(['.pdf', '.docx', '.doc', '.txt'])

        return sorted(list(set(extensions)))