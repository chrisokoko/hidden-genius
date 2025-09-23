"""
Shared types and data classes for the document processing service.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum


class FileType(Enum):
    """Supported file types for processing."""
    AUDIO = "audio"
    PDF = "pdf"
    DOCX = "docx"
    UNKNOWN = "unknown"


@dataclass
class ProcessorResult:
    """
    Result object returned by all processors.

    Attributes:
        success: Whether processing was successful
        text: Extracted text content
        source_file: Path to the input file
        output_file: Path to the output file (if saved)
        processor_type: Type of processor used
        processing_time: Time taken to process in seconds
        error_message: Error message if processing failed
    """
    success: bool
    text: str
    source_file: Path
    output_file: Optional[Path]
    processor_type: str
    processing_time: float
    error_message: Optional[str] = None