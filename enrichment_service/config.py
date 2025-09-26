"""
Configuration for enrichment service using Pydantic settings.
"""

from pydantic_settings import BaseSettings


class EnrichmentConfig(BaseSettings):
    """Configuration for enrichment service with type validation."""

    # Input/Output paths
    fingerprints_dir: str = "data/librarie/01_fingerprints"

    # Processing settings
    max_retries: int = 3

    class Config:
        env_file = ".env"
        env_prefix = "ENRICHMENT_"
        extra = "ignore"