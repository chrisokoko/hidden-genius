"""
Configuration for the embedding service using Pydantic Settings.

Handles environment variables, type validation, and default values.
"""

from pathlib import Path
from typing import Dict, Any
from pydantic import validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Force load from specific .env file, overriding any existing env vars
load_dotenv(".env", override=True)


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding generation with type validation."""

    # OpenAI settings
    openai_api_key: str
    model: str = "text-embedding-3-small"

    # Processing settings
    rate_limit_delay: float = 0.1
    max_retries: int = 3

    # Embedding strategies - different field combinations for different purposes
    embedding_strategies: Dict[str, Dict[str, Any]] = {
        "default": {
            "description": "Current default extraction (backward compatibility)",
            "fields": [
                "title",
                "essence",
                "core_exploration.central_question",
                "core_exploration.key_tension",
                "core_exploration.breakthrough_moment",
                "core_exploration.edge_of_understanding",
                "conceptual_dna"
            ]
        },
        "core_questions": {
            "description": "Focus on central questions and tensions",
            "fields": [
                "core_exploration.central_question",
                "core_exploration.key_tension"
            ]
        },
        "similarity": {
            "description": "Fields useful for similarity matching",
            "fields": [
                "conceptual_dna",
                "essence",
                "tags"
            ]
        },
        "themes": {
            "description": "Thematic categorization fields",
            "fields": [
                "tags",
                "keywords",
                "core_exploration.central_question"
            ]
        }
    }

    # Active strategy (can be overridden via CLI)
    active_strategy: str = "default"

    @validator('rate_limit_delay')
    def validate_rate_limit(cls, v):
        """Ensure rate limit delay is reasonable."""
        if v < 0:
            raise ValueError('rate_limit_delay must be non-negative')
        if v > 5.0:
            raise ValueError('rate_limit_delay should not exceed 5 seconds')
        return v

    @validator('max_retries')
    def validate_max_retries(cls, v):
        """Ensure max retries is reasonable."""
        if v < 0:
            raise ValueError('max_retries must be non-negative')
        if v > 10:
            raise ValueError('max_retries should not exceed 10')
        return v

    @validator('active_strategy')
    def validate_active_strategy(cls, v, values):
        """Ensure active strategy exists in embedding strategies."""
        if 'embedding_strategies' in values:
            strategies = values['embedding_strategies']
            if v not in strategies:
                available = list(strategies.keys())
                raise ValueError(f"Strategy '{v}' not found. Available strategies: {available}")
        return v

    @validator('embedding_strategies')
    def validate_embedding_strategies(cls, v):
        """Ensure embedding strategies are properly structured."""
        if not isinstance(v, dict):
            raise ValueError('embedding_strategies must be a dictionary')

        if not v:
            raise ValueError('At least one embedding strategy must be defined')

        for strategy_name, strategy in v.items():
            if not isinstance(strategy, dict):
                raise ValueError(f"Strategy '{strategy_name}' must be a dictionary")

            if 'fields' not in strategy:
                raise ValueError(f"Strategy '{strategy_name}' must have 'fields' list")

            if not isinstance(strategy['fields'], list):
                raise ValueError(f"Strategy '{strategy_name}' fields must be a list")

            if not strategy['fields']:
                raise ValueError(f"Strategy '{strategy_name}' must have at least one field")

        return v

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env