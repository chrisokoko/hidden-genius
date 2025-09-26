"""
Text file processor that passes through .txt file content.

This processor simply reads text files and returns their content without any processing.
It maintains consistency with the factory pattern while providing direct text access.
"""

import logging
import time
from pathlib import Path

from ..core.base import BaseProcessor
from ..core.types import ProcessorResult


# Setup logging
logger = logging.getLogger(__name__)


class TxtProcessor(BaseProcessor):
    """Handles text files by reading and passing through their content."""

    SUPPORTED_EXTENSIONS = {'.txt'}

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        return file_path.suffix.lower() == '.txt'

    def process(self, file_path: Path) -> ProcessorResult:
        """Process text file by reading its content."""
        start_time = time.time()

        try:
            # Get file info
            file_size = file_path.stat().st_size
            logger.info(f"Reading text file: {file_path.name} ({file_size:,} bytes)")

            # Read file content with UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                logger.warning(f"UTF-8 decode failed for {file_path.name}, trying latin-1")
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            # Ensure we have some content
            if not text.strip():
                logger.warning(f"Empty text file: {file_path.name}")
                text = "[WARNING: Text file appears to be empty]"

            duration = time.time() - start_time
            logger.info(f"✅ Completed: {file_path.name} ({duration:.3f}s, {len(text)} chars)")

            return ProcessorResult(
                success=True,
                text=text,
                source_file=file_path,
                processor_type="txt",
                processing_time=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Failed to read text file: {str(e)}"
            logger.error(f"❌ Error processing {file_path.name}: {error_msg}")

            return ProcessorResult(
                success=False,
                text="",
                source_file=file_path,
                processor_type="txt",
                processing_time=duration,
                error_message=error_msg
            )