"""
Abstract base class for all document processors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from .types import ProcessorResult


class BaseProcessor(ABC):
    """
    Abstract base class that all document processors must implement.

    This ensures a consistent interface across all file type processors,
    making the factory pattern work seamlessly.
    """

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this processor can handle the file, False otherwise
        """
        pass

    @abstractmethod
    def process(self, file_path: Path, output_dir: Path) -> ProcessorResult:
        """
        Process the file and extract text.

        Args:
            file_path: Path to the file to process
            output_dir: Directory where the output text file should be saved

        Returns:
            ProcessorResult object containing the extracted text and metadata
        """
        pass