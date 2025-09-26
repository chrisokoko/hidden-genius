"""
Configuration for clustering service using Pydantic settings.
"""

from typing import Dict, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class ClusteringConfig(BaseSettings):
    """Configuration for clustering service with type validation."""

    # Core clustering parameters
    min_clusters: int = 3
    target_count: int = 200

    # Algorithm settings (tunable but with good defaults)
    distance_metric: str = "cosine"
    linkage_method: str = "ward"

    # Quality score weights (advanced tuning)
    quality_weights: Dict[str, float] = {
        "silhouette": 0.35,
        "balance": 0.20,
        "separation": 0.25,
        "stability": 0.20
    }

    # Peak detection parameters (signal processing)
    # Made more sensitive to detect more natural breaks
    peak_detection: Dict[str, Optional[float]] = {
        "height": None,  # No minimum height - find all peaks
        "distance": 3.0,  # Allow closer peaks (was 5)
        "prominence": 0.005,  # More sensitive (was 0.01)
        "width": 1.0,  # Minimum width for peaks
        "rel_height": 0.5  # For measuring width at half prominence
    }

    @validator('min_clusters')
    def validate_min_clusters(cls, v):
        """Ensure min_clusters is reasonable."""
        if v < 2:
            raise ValueError('min_clusters must be at least 2')
        if v > 100:
            raise ValueError('min_clusters should not exceed 100')
        return v

    @validator('target_count')
    def validate_target_count(cls, v):
        """Ensure target_count is reasonable."""
        if v < 10:
            raise ValueError('target_count must be at least 10')
        if v > 1000:
            raise ValueError('target_count should not exceed 1000')
        return v

    @validator('quality_weights')
    def validate_quality_weights(cls, v):
        """Ensure quality weights sum approximately to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f'quality_weights should sum to 1.0, got {total}')
        return v

    class Config:
        env_file = ".env"
        env_prefix = "CLUSTERING_"
        extra = "ignore"  # Ignore extra fields from .env