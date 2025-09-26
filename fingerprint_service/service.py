"""
Semantic Fingerprinting Service

Processes transcripts through Claude to generate semantic fingerprints.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union

from claude_service import ClaudeService

# Setup logging
logger = logging.getLogger(__name__)

def load_semantic_prompt() -> str:
    """Load the semantic fingerprinting prompt from file."""
    prompt_file = Path(__file__).parent / "prompts" / "semantic_fingerprint.txt"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def process_transcripts(input_path: Union[str, Path], output_dir: Union[str, Path], max_workers: int = 1) -> Dict:
    """
    Process transcript(s) to generate semantic fingerprints using Claude service.

    Args:
        input_path: Single text file or directory containing text files
        output_dir: Directory where .json fingerprints will be saved
        max_workers: Number of concurrent threads (default: 1)

    Returns:
        Dict with processing stats: {'processed': int, 'failed': int, 'skipped': int}

    Raises:
        ValueError: If input_path doesn't exist or output_dir can't be created
        RuntimeError: If Claude API fails
    """
    print("🧬 Starting Semantic Fingerprint Processing with Enhanced Claude Service")
    print("=" * 75)

    # Convert to Path objects
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Validate input exists
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    # Initialize Claude service
    claude_service = ClaudeService()

    # Load semantic prompt
    semantic_prompt = load_semantic_prompt()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files to process
    if input_path.is_file():
        # Single file
        transcript_files = [input_path]
        transcript_dir = input_path.parent
    else:
        # Directory
        transcript_dir = input_path
        transcript_files = []

        # Get all files in directory
        for transcript_path in transcript_dir.rglob("*"):
            if not transcript_path.is_file():
                continue

            relative_path = transcript_path.relative_to(transcript_dir)
            output_path = output_dir / relative_path.with_suffix('.json')

            if not output_path.exists():
                # Check if file has content
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if len(content) >= 20:
                        transcript_files.append(transcript_path)
                except Exception:
                    pass

    logger.info(f"Found {len(transcript_files)} files needing processing")

    if not transcript_files:
        return {'processed': 0, 'failed': 0, 'skipped': 0}

    # Prepare data for processing
    batch_data = []
    file_mappings = []

    for file_path in transcript_files:
        relative_path = file_path.relative_to(transcript_dir)
        output_path = output_dir / relative_path.with_suffix('.json')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            batch_data.append({"content": content})
            file_mappings.append({
                'file_path': file_path,
                'output_path': output_path,
                'relative_path': relative_path
            })
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if not batch_data:
        return {'processed': 0, 'failed': 0, 'skipped': 0}

    print(f"\n📄 Processing {len(batch_data)} documents with Claude service...")

    # Use Claude service with list interface (automatically optimized)
    response = claude_service.send_message(
        prompt=semantic_prompt,
        data_list=batch_data,
        model="claude-3-5-sonnet-20241022",
        temperature=0.3
    )

    # Process the single consolidated response
    processed = 0
    failed = 0

    try:
        # Get the consolidated response content
        response_content = response.get_content()

        # Split the response by double newlines (how responses were joined)
        individual_responses = response_content.split('\n\n')

        if len(individual_responses) != len(file_mappings):
            logger.warning(f"Expected {len(file_mappings)} responses, got {len(individual_responses)}")

        # Process each individual response
        for i, file_info in enumerate(file_mappings):
            relative_path = file_info['relative_path']
            output_path = file_info['output_path']

            try:
                # Get the response for this file
                if i < len(individual_responses):
                    individual_response = individual_responses[i]
                else:
                    logger.error(f"No response found for {relative_path}")
                    failed += 1
                    continue

                # Check for error markers
                if individual_response.startswith("[ERROR:"):
                    logger.error(f"Claude processing failed for {relative_path}: {individual_response}")
                    failed += 1
                    continue

                # Parse JSON from Claude response
                json_start = individual_response.find('{')
                json_end = individual_response.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = individual_response[json_start:json_end]
                    fingerprint = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found")

                # Add metadata (using aggregated metrics from the full response)
                metrics = response.get_metrics()
                fingerprint['metadata'] = {
                    'source_file': str(relative_path),
                    'processed_at': datetime.now().isoformat(),
                    'model': metrics.get('model', 'claude-3-5-sonnet-20241022'),
                    'cache_creation_tokens': metrics.get('cache_write_tokens', 0),
                    'cache_read_tokens': metrics.get('cache_read_tokens', 0),
                    'total_tokens': metrics.get('total_tokens', 0),
                    'estimated_cost': metrics.get('estimated_cost', 0)
                }

                # Save fingerprint
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(fingerprint, f, indent=2)

                logger.info(f"✅ Saved: {relative_path}")
                processed += 1

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON parse error for {relative_path}: {e}")
                logger.error(f"Response sample: {individual_response[:500]}...")
                failed += 1
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
                failed += 1

    except Exception as e:
        logger.error(f"Error processing consolidated response: {e}")
        failed = len(file_mappings)

    print(f"\n✅ Processing complete!")
    print(f"✅ Processed: {processed}")
    print(f"❌ Failed: {failed}")

    return {
        'processed': processed,
        'failed': failed,
        'skipped': 0
    }