"""
Configuration for naming service using Pydantic settings.
"""

from pydantic_settings import BaseSettings


class NamingConfig(BaseSettings):
    """Configuration for naming service with type validation."""

    # Prompts
    prompts_dir: str = "naming_service/prompts"

    # Claude API settings
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.3

    # Processing settings
    max_essences_per_cluster: int = 20
    max_retries: int = 3

    class Config:
        env_file = ".env"
        env_prefix = "NAMING_"
        extra = "ignore"