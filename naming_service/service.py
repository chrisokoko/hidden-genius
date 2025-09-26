"""
Main naming service that generates Claude-powered cluster names and descriptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from claude_service import ClaudeService, ClaudeResponse
from .config import NamingConfig


logger = logging.getLogger(__name__)


class NamingService:
    """
    Naming service that generates Claude-powered names and descriptions for clusters.
    """

    def __init__(self, config: Optional[NamingConfig] = None):
        """Initialize the naming service."""
        self._config = config or NamingConfig()
        self._claude = ClaudeService()
        self._total_metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "total_execution_time": 0,
            "total_cost": 0,
            "api_calls": 0
        }

    def load_naming_prompt(self) -> str:
        """Load the cluster naming prompt template."""
        prompts_dir = Path(self._config.prompts_dir)
        prompt_path = prompts_dir / "cluster_naming.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Cluster naming prompt not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def validate_input_data(self, data: Dict) -> List[Dict]:
        """Validate input data structure and return clusters."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object")

        clusters = data.get('clusters')
        if not clusters:
            raise ValueError("Input must contain 'clusters' array")

        if not isinstance(clusters, list):
            raise ValueError("'clusters' must be an array")

        for i, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                raise ValueError(f"Cluster {i} must be an object")

            if 'cluster_id' not in cluster:
                raise ValueError(f"Cluster {i} missing required 'cluster_id' field")

            if 'files' not in cluster:
                raise ValueError(f"Cluster {cluster['cluster_id']} missing required 'files' field")

            if not isinstance(cluster['files'], list):
                raise ValueError(f"Cluster {cluster['cluster_id']} 'files' must be an array")

        return clusters

    def generate_cluster_name(self, cluster: Dict) -> tuple[Dict[str, str], ClaudeResponse]:
        """Generate name and description for a cluster using Claude."""
        cluster_id = cluster['cluster_id']
        files = cluster.get('files', [])

        # Check if cluster has any files with essence data
        files_with_essence = [f for f in files if f.get("essence")]

        if not files_with_essence:
            # Create a mock response for empty clusters
            empty_response = ClaudeResponse(
                content="No essence data - no Claude call made",
                metrics={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "total_tokens": 0,
                    "execution_time": 0,
                    "estimated_cost": 0,
                    "model": "n/a"
                }
            )
            return {
                "cluster_name": f"Empty Cluster ({len(files)} files)",
                "cluster_description": "No essence data available for this cluster."
            }, empty_response

        # Extract and format data for Claude - only what the prompt needs
        cluster_data = {
            "cluster_id": cluster_id,
            "files": []
        }

        # Extract only the fields needed: filename, title, essence
        for file_data in files_with_essence:
            cluster_data["files"].append({
                "filename": file_data.get("filename", ""),
                "title": file_data.get("title", ""),
                "essence": file_data.get("essence", "")
            })

        # Load prompt template
        prompt_template = self.load_naming_prompt()

        # Send formatted cluster data to Claude
        try:
            claude_response = self._claude.send_message(
                prompt=prompt_template,
                data_list=[cluster_data],
                model=self._config.claude_model,
                temperature=self._config.claude_temperature
            )

            # Extract content for parsing
            response = claude_response.get_content()

            # Log metrics from response
            metrics = claude_response.get_metrics()
            logger.info(f"📊 Claude API call metrics:")
            logger.info(f"   Input tokens: {metrics['input_tokens']}")
            logger.info(f"   Output tokens: {metrics['output_tokens']}")
            logger.info(f"   Total tokens: {metrics['total_tokens']}")
            logger.info(f"   Execution time: {metrics['execution_time']}s")
            logger.info(f"   Estimated cost: ${metrics['estimated_cost']}")
            if metrics['cache_read_tokens'] > 0:
                logger.info(f"   Cache read tokens: {metrics['cache_read_tokens']}")
            if metrics['cache_write_tokens'] > 0:
                logger.info(f"   Cache write tokens: {metrics['cache_write_tokens']}")

            # Update total metrics
            self._total_metrics["total_input_tokens"] += metrics['input_tokens']
            self._total_metrics["total_output_tokens"] += metrics['output_tokens']
            self._total_metrics["total_cache_read_tokens"] += metrics['cache_read_tokens']
            self._total_metrics["total_cache_write_tokens"] += metrics['cache_write_tokens']
            self._total_metrics["total_execution_time"] += metrics['execution_time']
            self._total_metrics["total_cost"] += metrics['estimated_cost']
            self._total_metrics["api_calls"] += 1

            # Clean up the response and parse JSON
            clean_response = response.strip()

            # Extract JSON from response if it's wrapped in other text
            if "```json" in clean_response:
                start = clean_response.find("```json") + 7
                end = clean_response.find("```", start)
                clean_response = clean_response[start:end].strip()
            elif "{" in clean_response and "}" in clean_response:
                start = clean_response.find("{")
                end = clean_response.rfind("}") + 1
                clean_response = clean_response[start:end]

            # Additional cleaning - remove any leading/trailing whitespace and newlines
            clean_response = clean_response.strip()

            result = json.loads(clean_response)
            return {
                "cluster_name": result.get("cluster_name", result.get("name", "Unnamed Cluster")),
                "cluster_description": result.get("cluster_description", result.get("description", "No description available."))
            }, claude_response

        except Exception as e:
            logger.error(f"Error generating cluster name for {cluster_id}: {e}")
            logger.error(f"Raw response was: {repr(response) if 'response' in locals() else 'No response'}")

            # Return error response
            error_response = claude_response if 'claude_response' in locals() else ClaudeResponse(
                content=f"Error - {str(e)}",
                metrics={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "total_tokens": 0,
                    "execution_time": 0,
                    "estimated_cost": 0,
                    "model": "error"
                }
            )
            return {
                "cluster_name": f"Cluster {cluster_id}",
                "cluster_description": "Error generating description from Claude."
            }, error_response

    def process_clusters(self, clusters: List[Dict]) -> tuple[List[Dict], List[tuple[str, ClaudeResponse]]]:
        """Process all clusters and generate names."""
        named_clusters = []
        raw_responses = []

        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            files_with_essence = [f for f in cluster.get('files', []) if f.get('essence')]

            logger.info(f"Generating name for cluster {cluster_id} "
                       f"({len(files_with_essence)} essences)")

            naming_result, raw_response = self.generate_cluster_name(cluster)

            # Create clean cluster output with only essential fields
            named_cluster = {
                "cluster_id": cluster_id,
                "cluster_name": naming_result["cluster_name"],
                "cluster_description": naming_result["cluster_description"]
            }

            named_clusters.append(named_cluster)
            raw_responses.append((cluster_id, raw_response))

        logger.info(f"Generated names for {len(named_clusters)} clusters")
        return named_clusters, raw_responses

    def save_raw_responses(self, raw_responses: List[tuple[str, ClaudeResponse]],
                          source_file: str, output_file: Path) -> None:
        """Save raw Claude responses."""
        # Create raw filename from output file
        raw_file = output_file.parent / f"{output_file.stem}_raw.json"

        # Use ClaudeService's consolidation method
        consolidated_data = ClaudeService.consolidate_responses(
            raw_responses,
            source_file=source_file
        )

        # Clean up the response content to only include essential cluster info
        for response_item in consolidated_data["responses"]:
            try:
                content = response_item["content"]
                if content.startswith("{") and content.endswith("}"):
                    cluster_result = json.loads(content)
                    # Keep only essential fields
                    cleaned_result = {
                        "cluster_id": cluster_result.get("cluster_id"),
                        "cluster_name": cluster_result.get("cluster_name"),
                        "cluster_description": cluster_result.get("cluster_description")
                    }
                    response_item["content"] = json.dumps(cleaned_result, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, keep as-is (could be error message)
                pass

        # Write consolidated raw responses as JSON
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2)

        logger.info(f"💾 Saved raw responses to: {raw_file}")

    def save_final_output(self, named_clusters: List[Dict], output_file: Path) -> None:
        """Save final clean output with only essential cluster data."""
        output_data = {
            "clusters": named_clusters
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved final output to: {output_file}")

    def process_file(self, input_file: Path, output_file: Path) -> Dict[str, int]:
        """Process a single cluster file."""
        logger.info(f"🏷️  Processing file: {input_file}")

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load and validate input data
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            clusters = self.validate_input_data(input_data)
            logger.info(f"Loaded {len(clusters)} clusters from {input_file}")

            # Process clusters
            named_clusters, raw_responses = self.process_clusters(clusters)

            # Save outputs
            self.save_raw_responses(raw_responses, str(input_file), output_file)
            self.save_final_output(named_clusters, output_file)

            # Log summary
            self._log_summary(named_clusters)
            self._log_total_api_metrics()

            return {'processed': len(named_clusters), 'failed': 0, 'skipped': 0}

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'processed': 0, 'failed': 1, 'skipped': 0}

    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Process all JSON files in a directory."""
        logger.info(f"🏷️  Processing directory: {input_dir}")

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return {'processed': 0, 'failed': 0, 'skipped': 0}

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        total_results = {'processed': 0, 'failed': 0, 'skipped': 0}

        for input_file in json_files:
            output_file = output_dir / input_file.name
            logger.info(f"Processing: {input_file.name}")

            result = self.process_file(input_file, output_file)

            # Aggregate results
            for key in total_results:
                total_results[key] += result[key]

        logger.info(f"Directory processing complete: {total_results}")
        return total_results

    def _log_summary(self, named_clusters: List[Dict]) -> None:
        """Log summary of naming results."""
        logger.info("\n📋 NAMING SUMMARY:")

        total_clusters = len(named_clusters)
        clusters_with_names = sum(1 for cluster in named_clusters
                                if cluster.get('cluster_name') and
                                not cluster['cluster_name'].startswith('Cluster '))

        logger.info(f"   Total Clusters: {total_clusters}")
        logger.info(f"   Successfully Named: {clusters_with_names}")
        if total_clusters > 0:
            logger.info(f"   Success Rate: {clusters_with_names/total_clusters*100:.1f}%")

    def _log_total_api_metrics(self) -> None:
        """Log total API metrics summary."""
        logger.info("\n💰 TOTAL CLAUDE API METRICS:")
        logger.info(f"   API Calls: {self._total_metrics['api_calls']}")
        logger.info(f"   Total Input Tokens: {self._total_metrics['total_input_tokens']:,}")
        logger.info(f"   Total Output Tokens: {self._total_metrics['total_output_tokens']:,}")
        total_tokens = self._total_metrics['total_input_tokens'] + self._total_metrics['total_output_tokens']
        logger.info(f"   Total Tokens Used: {total_tokens:,}")

        if self._total_metrics['total_cache_read_tokens'] > 0:
            logger.info(f"   Total Cache Read Tokens: {self._total_metrics['total_cache_read_tokens']:,}")
        if self._total_metrics['total_cache_write_tokens'] > 0:
            logger.info(f"   Total Cache Write Tokens: {self._total_metrics['total_cache_write_tokens']:,}")

        logger.info(f"   Total Execution Time: {self._total_metrics['total_execution_time']:.2f}s")
        logger.info(f"   TOTAL ESTIMATED COST: ${self._total_metrics['total_cost']:.4f}")

        if self._total_metrics['api_calls'] > 0:
            avg_time = self._total_metrics['total_execution_time'] / self._total_metrics['api_calls']
            avg_cost = self._total_metrics['total_cost'] / self._total_metrics['api_calls']
            logger.info(f"   Average Time per Call: {avg_time:.2f}s")
            logger.info(f"   Average Cost per Call: ${avg_cost:.4f}")