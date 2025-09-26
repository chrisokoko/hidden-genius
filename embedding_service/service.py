"""
Embedding Service - Modern class-based architecture in single file.

Converts semantic fingerprints to embedding vectors using OpenAI API.
Organized following industry best practices for services of this size.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PROCESSING STATISTICS
# =============================================================================

class ProcessingStats:
    """Track processing statistics."""

    def __init__(self):
        self.processed = 0
        self.failed = 0
        self.skipped = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert stats to dictionary format."""
        return {
            'processed': self.processed,
            'failed': self.failed,
            'skipped': self.skipped
        }

    def __str__(self) -> str:
        """String representation for logging."""
        return f"Processed: {self.processed}, Failed: {self.failed}, Skipped: {self.skipped}"


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

class FingerprintExtractor:
    """Extracts embedding text from semantic fingerprint data."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize extractor with configuration.

        Args:
            config: Embedding configuration with extraction field settings
        """
        self.config = config

    def extract_embedding_text(self, fingerprint: Dict) -> str:
        """Extract text from fingerprint using configured strategy.

        Args:
            fingerprint: Parsed fingerprint JSON data

        Returns:
            Formatted text string ready for embedding

        Raises:
            ValueError: If strategy not found or no extractable text found
        """
        # Get active strategy
        strategy_name = self.config.active_strategy
        if strategy_name not in self.config.embedding_strategies:
            available = list(self.config.embedding_strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {available}")

        strategy = self.config.embedding_strategies[strategy_name]
        text_parts = []

        # Extract fields specified in strategy
        for field_path in strategy["fields"]:
            try:
                field_text = self._extract_field(fingerprint, field_path)
                if field_text:
                    text_parts.append(field_text)
            except ValueError as e:
                raise ValueError(f"Strategy '{strategy_name}' failed: {e}")

        if not text_parts:
            raise ValueError(f"No extractable text found using strategy '{strategy_name}'")

        return " | ".join(text_parts)

    def _extract_field(self, fingerprint: Dict, field_path: str) -> str:
        """Extract a field from fingerprint using dot notation path.

        Args:
            fingerprint: Parsed fingerprint JSON data
            field_path: Field path (e.g., 'essence' or 'core_exploration.central_question')

        Returns:
            Formatted field text

        Raises:
            ValueError: If field not found or empty
        """
        # Split the field path (e.g., "core_exploration.central_question")
        path_parts = field_path.split('.')

        # Navigate through the nested structure
        current = fingerprint
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Field '{field_path}' not found in fingerprint")
            current = current[part]

        # Handle different data types
        if isinstance(current, list):
            if not current:
                raise ValueError(f"Field '{field_path}' is empty")
            # For lists, join with spaces
            return " ".join(str(item) for item in current)

        elif isinstance(current, str):
            if not current.strip():
                raise ValueError(f"Field '{field_path}' is empty")
            return current.strip()

        elif isinstance(current, dict):
            # For dict, we might want to extract all values
            if not current:
                raise ValueError(f"Field '{field_path}' is empty")
            # Join all non-empty string values
            values = [str(v) for v in current.values() if v and str(v).strip()]
            if not values:
                raise ValueError(f"Field '{field_path}' contains no extractable text")
            return " ".join(values)

        else:
            # Convert other types to string
            result = str(current).strip()
            if not result:
                raise ValueError(f"Field '{field_path}' is empty")
            return result

# =============================================================================
# OPENAI API CLIENT
# =============================================================================

class OpenAIClient:
    """Handles OpenAI API calls for embedding generation."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI client with configuration.

        Args:
            config: Embedding configuration with API settings

        Raises:
            ValueError: If OpenAI API key is not provided
        """
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")

        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding vector

        Raises:
            RuntimeError: If API call fails after all retries
            ValueError: If text is empty or too long
        """
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        # Log the request (truncated for readability)
        truncated_text = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"Generating embedding for text: {truncated_text}")

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    input=text,
                    model=self.config.model
                )

                embedding_vector = response.data[0].embedding

                logger.debug(f"✅ Generated embedding: {len(embedding_vector)} dimensions")
                return embedding_vector

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    retry_delay = self.config.rate_limit_delay * (attempt + 1)
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

        # All retries exhausted
        raise RuntimeError(
            f"Failed to generate embedding after {self.config.max_retries + 1} attempts. "
            f"Last error: {last_exception}"
        )

    def apply_rate_limit(self) -> None:
        """Apply rate limiting delay between API calls."""
        if self.config.rate_limit_delay > 0:
            logger.debug(f"Rate limiting: sleeping {self.config.rate_limit_delay}s")
            time.sleep(self.config.rate_limit_delay)

    def get_model_info(self) -> dict:
        """Get information about the current embedding model.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.config.model,
            "expected_dimensions": self._get_expected_dimensions(),
            "max_input_tokens": self._get_max_input_tokens()
        }

    def _get_expected_dimensions(self) -> int:
        """Get expected dimensions for the current model.

        Returns:
            Number of dimensions for the embedding model
        """
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions_map.get(self.config.model, 1536)

    def _get_max_input_tokens(self) -> int:
        """Get maximum input tokens for the current model.

        Returns:
            Maximum number of input tokens
        """
        # Most OpenAI embedding models support 8192 tokens
        return 8192


# =============================================================================
# FILE HANDLING
# =============================================================================

class FileHandler:
    """Handles file I/O operations for the embedding service."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize file handler with configuration.

        Args:
            config: Embedding configuration
        """
        self.config = config

    def collect_fingerprint_files(self, input_path: Union[str, Path]) -> List[Path]:
        """Collect fingerprint files to process.

        Args:
            input_path: Single file or directory path

        Returns:
            List of fingerprint file paths to process

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If no valid fingerprint files found
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            if not self._is_json_file(input_path):
                raise ValueError(f"Input file must be a .json file: {input_path}")
            return [input_path]

        if input_path.is_dir():
            fingerprint_files = list(input_path.rglob("*.json"))
            if not fingerprint_files:
                raise ValueError(f"No .json files found in directory: {input_path}")

            logger.info(f"Found {len(fingerprint_files)} fingerprint files")
            return fingerprint_files

        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    def load_fingerprint(self, file_path: Path) -> Dict:
        """Load and parse a fingerprint JSON file.

        Args:
            file_path: Path to fingerprint JSON file

        Returns:
            Parsed fingerprint data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or missing required fields
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Fingerprint file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                fingerprint = json.load(f)

            # Basic validation - ensure it's a dictionary
            if not isinstance(fingerprint, dict):
                raise ValueError("Fingerprint must be a JSON object")

            logger.debug(f"✅ Loaded fingerprint: {file_path.name}")
            return fingerprint

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load fingerprint {file_path}: {e}")

    def save_embedding(self, embedding_vector: List[float], embedding_text: str,
                      source_path: Path, output_dir: Path) -> Path:
        """Save embedding data to JSON file.

        Args:
            embedding_vector: Generated embedding vector
            embedding_text: Text that was embedded
            source_path: Path to source fingerprint file
            output_dir: Output directory for embedding

        Returns:
            Path where embedding was saved

        Raises:
            OSError: If unable to create directory or write file
        """
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use flat structure - just filename in output directory
        output_path = output_dir / source_path.name

        # Create embedding data structure
        embedding_data = {
            "embedding_vector": embedding_vector,
            "embedding_text": embedding_text,
            "model": self.config.model,
            "dimensions": len(embedding_vector),
            "source_fingerprint": str(source_path.absolute()),
            "processed_at": datetime.now().isoformat()
        }

        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, indent=2)

            logger.debug(f"✅ Saved embedding: {output_path.name}")
            return output_path

        except Exception as e:
            raise OSError(f"Failed to save embedding to {output_path}: {e}")

    def embedding_exists(self, source_path: Path, output_dir: Path) -> bool:
        """Check if embedding already exists for a fingerprint.

        Args:
            source_path: Path to source fingerprint file
            output_dir: Output directory to check

        Returns:
            True if embedding file already exists
        """
        output_path = output_dir / source_path.name
        exists = output_path.exists()

        if exists:
            logger.debug(f"Embedding exists: {output_path.name}")

        return exists

    def validate_input_path(self, input_path: Union[str, Path]) -> Path:
        """Validate and convert input path.

        Args:
            input_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If path is invalid type for processing
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file() and not self._is_json_file(input_path):
            raise ValueError(f"Input file must be a .json file: {input_path}")

        return input_path

    def _is_json_file(self, file_path: Path) -> bool:
        """Check if file has .json extension.

        Args:
            file_path: File path to check

        Returns:
            True if file has .json extension
        """
        return file_path.suffix.lower() == '.json'


# =============================================================================
# MAIN EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    """Orchestrates embedding generation from semantic fingerprints."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize service with configuration and components.

        Args:
            config: Validated embedding configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.extractor = FingerprintExtractor(config)
        self.client = OpenAIClient(config)
        self.file_handler = FileHandler(config)

        logger.info(f"Initialized EmbeddingService with model: {config.model}")

    def process_directory(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, int]:
        """Process fingerprint file(s) to generate embeddings.

        Args:
            input_path: Single fingerprint file or directory containing fingerprints
            output_dir: Directory where embeddings will be saved

        Returns:
            Dictionary with processing statistics

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If input format is invalid
        """
        print("🔮 Starting Embedding Processing")
        print("=" * 60)

        # Validate and prepare paths
        input_path = self.file_handler.validate_input_path(input_path)
        output_dir = Path(output_dir)

        # Collect files to process
        fingerprint_files = self.file_handler.collect_fingerprint_files(input_path)
        stats = ProcessingStats()

        logger.info(f"Processing {len(fingerprint_files)} fingerprint files")

        # Process each file
        for file_path in fingerprint_files:
            try:
                result = self._process_single_file(file_path, output_dir)
                if result == "processed":
                    stats.processed += 1
                elif result == "skipped":
                    stats.skipped += 1
                else:
                    stats.failed += 1

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                stats.failed += 1

        # Print final statistics
        self._print_final_stats(stats)
        return stats.to_dict()

    def process_file(self, file_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, int]:
        """Process a single fingerprint file.

        Args:
            file_path: Path to fingerprint JSON file
            output_dir: Directory where embedding will be saved

        Returns:
            Dictionary with processing statistics
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)

        stats = ProcessingStats()
        result = self._process_single_file(file_path, output_dir)

        if result == "processed":
            stats.processed += 1
        elif result == "skipped":
            stats.skipped += 1
        else:
            stats.failed += 1

        return stats.to_dict()

    def get_model_info(self) -> Dict:
        """Get information about the embedding model.

        Returns:
            Dictionary with model configuration and capabilities
        """
        return {
            "service_config": {
                "model": self.config.model,
                "rate_limit_delay": self.config.rate_limit_delay,
                "max_retries": self.config.max_retries
            },
            "extraction_fields": self.config.extraction_fields,
            "model_info": self.client.get_model_info()
        }

    def _process_single_file(self, file_path: Path, output_dir: Path) -> str:
        """Process a single fingerprint file to embedding.

        Args:
            file_path: Path to fingerprint file
            output_dir: Output directory

        Returns:
            Processing result: "processed", "skipped", or "failed"
        """
        # Check if already exists
        if self.file_handler.embedding_exists(file_path, output_dir):
            logger.info(f"Skipping existing: {file_path.name}")
            return "skipped"

        try:
            # Load fingerprint
            fingerprint = self.file_handler.load_fingerprint(file_path)

            # Extract text for embedding
            embedding_text = self.extractor.extract_embedding_text(fingerprint)

            logger.info(f"Processing embedding: {file_path.name}")

            # Generate embedding
            embedding_vector = self.client.generate_embedding(embedding_text)

            # Save embedding
            self.file_handler.save_embedding(
                embedding_vector=embedding_vector,
                embedding_text=embedding_text,
                source_path=file_path,
                output_dir=output_dir
            )

            logger.info(f"✅ Saved embedding: {file_path.name}")

            # Apply rate limiting
            self.client.apply_rate_limit()

            return "processed"

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            return "failed"

    def _print_final_stats(self, stats: ProcessingStats) -> None:
        """Print final processing statistics.

        Args:
            stats: Processing statistics to display
        """
        print(f"\n✅ Processed: {stats.processed}")
        print(f"⏭️ Skipped: {stats.skipped}")
        print(f"❌ Failed: {stats.failed}")

        if stats.failed > 0:
            print("\nSome files failed to process. Check logs for details.")


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def generate_embeddings(input_path: Union[str, Path], output_dir: Union[str, Path], config: EmbeddingConfig = None) -> Dict[str, int]:
    """Generate embeddings from fingerprint(s) - functional interface.

    Args:
        input_path: Single fingerprint file or directory
        output_dir: Output directory for embeddings
        config: Optional configuration override

    Returns:
        Dictionary with processing statistics

    Raises:
        ValueError: If configuration or input is invalid
    """
    try:
        if config is None:
            config = EmbeddingConfig()
        service = EmbeddingService(config)
        return service.process_directory(input_path, output_dir)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise