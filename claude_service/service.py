"""
Claude Service - Clean API interface for Claude AI operations
"""

import os
import json
import logging
import time
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

from .logger import ClaudeLogger

logger = logging.getLogger(__name__)


class ClaudeResponse:
    """Encapsulates a Claude API response with content and metadata."""

    def __init__(self, content: str, metrics: Dict[str, Any]):
        self._content = content
        self._metrics = metrics
        self._raw_data = {
            "content": content,
            **metrics
        }

    def get_content(self) -> str:
        """Get the response text content."""
        return self._content

    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics (tokens, cost, time, etc.)."""
        return self._metrics

    def get_raw(self) -> Dict[str, Any]:
        """Get the complete raw response data."""
        return self._raw_data

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get a summary of key metrics for logging."""
        return {
            "model": self._metrics["model"],
            "total_tokens": self._metrics["total_tokens"],
            "execution_time": self._metrics["execution_time"],
            "estimated_cost": self._metrics["estimated_cost"]
        }

    def to_json(self) -> str:
        """Convert the response to JSON string."""
        return json.dumps(self._raw_data, indent=2)


class ClaudeService:
    @staticmethod
    def consolidate_responses(responses: List[tuple[str, ClaudeResponse]],
                             source_file: str = None) -> Dict[str, Any]:
        """
        Consolidate multiple Claude responses into a single structured format.

        Args:
            responses: List of tuples (identifier, ClaudeResponse)
            source_file: Optional source file path

        Returns:
            Dictionary with consolidated responses and metrics
        """
        consolidated = {
            "timestamp": datetime.now().isoformat(),
            "source_file": source_file,
            "total_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_read_tokens": 0,
                "total_cache_write_tokens": 0,
                "total_tokens": 0,
                "total_execution_time": 0,
                "total_cost": 0,
                "api_calls": len(responses)
            },
            "responses": []
        }

        # Aggregate metrics and add individual responses
        for identifier, response in responses:
            metrics = response.get_metrics()

            # Update totals
            consolidated["total_metrics"]["total_input_tokens"] += metrics["input_tokens"]
            consolidated["total_metrics"]["total_output_tokens"] += metrics["output_tokens"]
            consolidated["total_metrics"]["total_cache_read_tokens"] += metrics["cache_read_tokens"]
            consolidated["total_metrics"]["total_cache_write_tokens"] += metrics["cache_write_tokens"]
            consolidated["total_metrics"]["total_tokens"] += metrics["total_tokens"]
            consolidated["total_metrics"]["total_execution_time"] += metrics["execution_time"]
            consolidated["total_metrics"]["total_cost"] += metrics["estimated_cost"]

            # Add individual response
            consolidated["responses"].append({
                "identifier": identifier,
                "content": response.get_content(),
                "metrics": metrics
            })

        # Round totals for cleaner output
        consolidated["total_metrics"]["total_execution_time"] = round(
            consolidated["total_metrics"]["total_execution_time"], 2
        )
        consolidated["total_metrics"]["total_cost"] = round(
            consolidated["total_metrics"]["total_cost"], 6
        )

        return consolidated

    def __init__(self, api_key: str = None):
        """
        Initialize the Claude service.

        Args:
            api_key: Claude API key (optional, will use CLAUDE_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY not found. Please provide it as a parameter or set the environment variable.")

        self.client = Anthropic(api_key=self.api_key)

        # Setup API usage logging
        self.logger = ClaudeLogger()
        logger.info(f"Claude API usage logging to: {self.logger.get_log_file_path()}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int,
                        cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        """Calculate the cost based on model and token usage."""
        # Pricing as of Oct 2024 (per million tokens)
        pricing = {
            "claude-3-5-sonnet-20241022": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,  # 25% more than base input
                "cache_read": 0.30    # 90% discount from base input
            },
            "claude-3-opus-20240229": {
                "input": 15.00,
                "output": 75.00,
                "cache_write": 18.75,
                "cache_read": 1.50
            },
            "claude-3-sonnet-20240229": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30
            },
            "claude-3-haiku-20240307": {
                "input": 0.25,
                "output": 1.25,
                "cache_write": 0.30,
                "cache_read": 0.03
            }
        }

        # Get pricing for the model, default to sonnet if not found
        model_pricing = pricing.get(model, pricing["claude-3-5-sonnet-20241022"])

        # Calculate costs in dollars (pricing is per million tokens)
        input_cost = (input_tokens * model_pricing["input"]) / 1_000_000
        output_cost = (output_tokens * model_pricing["output"]) / 1_000_000
        cache_write_cost = (cache_write_tokens * model_pricing["cache_write"]) / 1_000_000
        cache_read_cost = (cache_read_tokens * model_pricing["cache_read"]) / 1_000_000

        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

        return round(total_cost, 6)

    def send_message(self, prompt: str, data_list: List[Dict[str, Any]],
                     model: str = "claude-3-5-sonnet-20241022",
                     temperature: float = 0.3) -> ClaudeResponse:
        """
        Send a message to Claude with a prompt and list of data items.

        Args:
            prompt: The prompt template to use
            data_list: List of data dictionaries to process (required)
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
            temperature: Temperature for response (default: 0.3)

        Returns:
            ClaudeResponse object with consolidated results:
                - get_content(): Get the consolidated response text
                - get_metrics(): Get aggregated usage metrics
                - get_raw(): Get complete response data
        """
        # Validate that data_list is a list
        if not isinstance(data_list, list):
            raise ValueError("data_list must be a list")

        if len(data_list) == 0:
            raise ValueError("data_list cannot be empty")

        # Single item - process directly
        if len(data_list) == 1:
            return self._send_message_with_retry(
                prompt=prompt,
                data=data_list[0],
                model=model,
                temperature=temperature,
                enable_caching=False,
                max_retries=3
            )

        # Multiple items - use batch processing and consolidate results
        max_workers = min(len(data_list), 10)  # Cap at 10 workers

        responses = self._process_batch_internal(
            prompt=prompt,
            data_items=data_list,
            model=model,
            temperature=temperature,
            enable_caching=True,  # Use caching for batch processing
            max_workers=max_workers,
            max_retries=3
        )

        # Consolidate all responses into a single ClaudeResponse
        return self._consolidate_responses(responses, model)

    def _send_message_with_retry(self, prompt: str, data: Dict[str, Any] = None,
                                model: str = "claude-3-5-sonnet-20241022",
                                temperature: float = 0.3,
                                enable_caching: bool = False,
                                max_retries: int = 3) -> ClaudeResponse:
        """
        Internal method: Send a message to Claude with retry logic for rate limiting.
        """
        for attempt in range(max_retries):
            try:
                return self._send_message_with_caching(
                    prompt=prompt,
                    data=data,
                    model=model,
                    temperature=temperature,
                    enable_caching=enable_caching
                )
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                raise e

    def _send_message_with_caching(self, prompt: str, data: Dict[str, Any] = None,
                                  model: str = "claude-3-5-sonnet-20241022",
                                  temperature: float = 0.3,
                                  enable_caching: bool = False) -> ClaudeResponse:
        """
        Internal method: Send message with optional caching support.
        """
        # Format the prompt with data if provided
        final_prompt = prompt
        if data:
            # If data contains complex objects, convert to JSON
            formatted_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    formatted_data[key] = json.dumps(value, indent=2)
                else:
                    formatted_data[key] = value

            # Format the prompt with the data
            try:
                final_prompt = prompt.format(**formatted_data)
            except KeyError as e:
                # If template has placeholders not in data, append data as JSON
                logger.warning(f"Missing template key {e}, appending data as JSON")
                final_prompt = prompt + "\n\nData:\n" + json.dumps(data, indent=2)

        try:
            start_time = time.time()

            # Determine max_tokens based on model
            max_tokens = 4096 if "haiku" in model else 4000

            # Build message parameters
            message_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ]
            }

            # Add system message with cache control if caching is enabled
            if enable_caching:
                message_params["system"] = [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
                # Update user message to contain only the data
                if data:
                    formatted_data = {}
                    for key, value in data.items():
                        if isinstance(value, (dict, list)):
                            formatted_data[key] = json.dumps(value, indent=2)
                        else:
                            formatted_data[key] = value
                    message_params["messages"] = [
                        {
                            "role": "user",
                            "content": json.dumps(formatted_data, indent=2) if formatted_data else ""
                        }
                    ]

            response = self.client.messages.create(**message_params)

            execution_time = round(time.time() - start_time, 2)

            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            # Cache tokens - these are included in newer API responses
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
            cache_write_tokens = getattr(usage, 'cache_creation_input_tokens', 0)

            # Calculate total tokens
            total_tokens = input_tokens + output_tokens

            # Calculate estimated cost
            estimated_cost = self._calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens
            )

            content = response.content[0].text.strip()

            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "total_tokens": total_tokens,
                "execution_time": execution_time,
                "estimated_cost": estimated_cost,
                "model": model
            }

            # Create response object
            claude_response = ClaudeResponse(content, metrics)

            # Log API usage
            self.logger.log_api_call(final_prompt, claude_response.get_raw())

            return claude_response

        except Exception as e:
            logger.error(f"Error sending message to Claude: {e}")
            raise

    def _process_batch_internal(self,
                               prompt: str,
                               data_items: List[Dict[str, Any]],
                               model: str = "claude-3-5-sonnet-20241022",
                               temperature: float = 0.3,
                               enable_caching: bool = True,
                               max_workers: int = 1,
                               max_retries: int = 3,
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ClaudeResponse]:
        """
        Internal method: Process multiple data items with the same prompt using caching and threading.
        """
        if not data_items:
            return []

        logger.info(f"Internal batch processing: {len(data_items)} items with {max_workers} workers")

        # Create result list to maintain order
        results = [None] * len(data_items)

        # Thread-safe statistics
        stats = {'processed': 0, 'failed': 0}
        stats_lock = threading.Lock()

        def process_item(index: int, data: Dict[str, Any]) -> None:
            """Process a single item in the batch."""
            try:
                response = self._send_message_with_retry(
                    prompt=prompt,
                    data=data,
                    model=model,
                    temperature=temperature,
                    enable_caching=enable_caching,
                    max_retries=max_retries
                )
                results[index] = response

                with stats_lock:
                    stats['processed'] += 1
                    if progress_callback:
                        progress_callback(stats['processed'] + stats['failed'], len(data_items))

            except Exception as e:
                logger.error(f"Failed to process item {index}: {e}")
                results[index] = None

                with stats_lock:
                    stats['failed'] += 1
                    if progress_callback:
                        progress_callback(stats['processed'] + stats['failed'], len(data_items))

        # Process first item to create cache if caching is enabled
        if enable_caching:
            logger.debug("Creating cache with first item...")
            process_item(0, data_items[0])

            if len(data_items) > 1:
                logger.debug(f"Processing remaining {len(data_items)-1} items with {max_workers} workers...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit remaining tasks (skip first item)
                    futures = [
                        executor.submit(process_item, i, data)
                        for i, data in enumerate(data_items[1:], start=1)
                    ]

                    # Wait for all tasks to complete
                    for future in as_completed(futures):
                        try:
                            future.result()  # Get result or raise exception
                        except Exception as e:
                            logger.error(f"Batch task failed: {e}")
        else:
            # Process all items concurrently without cache optimization
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_item, i, data)
                    for i, data in enumerate(data_items)
                ]

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Batch task failed: {e}")

        logger.info(f"Internal batch processing complete: {stats['processed']} processed, {stats['failed']} failed")
        return results

    def _consolidate_responses(self, responses: List[ClaudeResponse], model: str) -> ClaudeResponse:
        """
        Internal method: Consolidate multiple Claude responses into a single response.
        """
        # Filter out None responses (failed requests)
        valid_responses = [r for r in responses if r is not None]

        if not valid_responses:
            raise RuntimeError("All requests failed")

        # Combine all content
        combined_content = []
        for i, response in enumerate(responses):
            if response is not None:
                combined_content.append(response.get_content())
            else:
                combined_content.append(f"[ERROR: Request {i} failed]")

        # Join all content - services can parse this as needed
        final_content = "\n\n".join(combined_content)

        # Aggregate metrics
        total_metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "execution_time": 0,
            "estimated_cost": 0,
            "model": model
        }

        for response in valid_responses:
            metrics = response.get_metrics()
            total_metrics["input_tokens"] += metrics.get("input_tokens", 0)
            total_metrics["output_tokens"] += metrics.get("output_tokens", 0)
            total_metrics["cache_read_tokens"] += metrics.get("cache_read_tokens", 0)
            total_metrics["cache_write_tokens"] += metrics.get("cache_write_tokens", 0)
            total_metrics["total_tokens"] += metrics.get("total_tokens", 0)
            total_metrics["execution_time"] += metrics.get("execution_time", 0)
            total_metrics["estimated_cost"] += metrics.get("estimated_cost", 0)

        # Round execution time and cost
        total_metrics["execution_time"] = round(total_metrics["execution_time"], 2)
        total_metrics["estimated_cost"] = round(total_metrics["estimated_cost"], 6)

        return ClaudeResponse(final_content, total_metrics)

