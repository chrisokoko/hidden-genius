"""
Utility functions for file type detection.
"""

from pathlib import Path
from typing import Optional


def detect_file_type(file_path: Path) -> Optional[str]:
    """
    Detect file type from extension.

    Args:
        file_path: Path to the file to check

    Returns:
        String indicating file type ('audio', 'pdf', 'docx') or None if unsupported
    """
    extension = file_path.suffix.lower()

    # Audio file extensions
    audio_extensions = {'.m4a', '.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma'}
    if extension in audio_extensions:
        return "audio"

    # PDF files
    elif extension == '.pdf':
        return "pdf"

    # Word documents
    elif extension in ['.docx', '.doc']:
        return "docx"

    # Unsupported type
    else:
        return None


def get_supported_extensions() -> list:
    """
    Get a list of all supported file extensions.

    Returns:
        List of supported file extensions including the dot
    """
    return [
        # Audio formats
        '.m4a', '.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma',
        # Document formats
        '.pdf', '.docx', '.doc'
    ]


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file is supported by any processor.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file type is supported, False otherwise
    """
    return detect_file_type(file_path) is not None