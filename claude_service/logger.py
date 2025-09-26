"""
Claude API usage logger - handles all logging for Claude API calls.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ClaudeLogger:
    """Handles logging of Claude API usage to files."""

    def __init__(self):
        """Initialize the logger with monthly log files."""
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup file logging for API usage tracking."""
        # Create logs directory
        service_dir = Path(__file__).parent
        logs_dir = service_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path with date
        log_filename = f"api_usage_{datetime.now().strftime('%Y%m')}.log"
        self.log_file_path = logs_dir / log_filename

        # Setup file handler
        self.api_logger = logging.getLogger("claude_api_usage")
        self.api_logger.setLevel(logging.INFO)

        # Only add handler if it doesn't exist
        if not self.api_logger.handlers:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)

            self.api_logger.addHandler(file_handler)
            self.api_logger.propagate = False

    def log_api_call(self, prompt: str, response_data: Dict[str, Any]) -> None:
        """Log an API call with full details."""
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": response_data["model"],
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "input_tokens": response_data["input_tokens"],
            "output_tokens": response_data["output_tokens"],
            "cache_read_tokens": response_data["cache_read_tokens"],
            "cache_write_tokens": response_data["cache_write_tokens"],
            "total_tokens": response_data["total_tokens"],
            "execution_time": response_data["execution_time"],
            "estimated_cost": response_data["estimated_cost"]
        }

        # Log as JSON for programmatic parsing
        self.api_logger.info(json.dumps(log_entry))

        # Log human-readable summary
        summary = (
            f"MODEL: {response_data['model']} | "
            f"TOKENS: {response_data['total_tokens']} "
            f"(in: {response_data['input_tokens']}, out: {response_data['output_tokens']}) | "
            f"TIME: {response_data['execution_time']}s | "
            f"COST: ${response_data['estimated_cost']:.4f}"
        )
        self.api_logger.info(f"SUMMARY: {summary}")
        self.api_logger.info("-" * 80)

    def get_log_file_path(self) -> Path:
        """Get the current log file path."""
        return self.log_file_path