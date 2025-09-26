"""
File I/O operations for the clustering service.

Handles loading embeddings from JSON files and saving clustering results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class IOHandler:
    """Handles all file I/O operations for the clustering service."""

    @staticmethod
    def load_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings and filenames from directory of JSON files.

        Args:
            embeddings_dir: Path to directory containing embedding files

        Returns:
            Tuple of (embeddings_array, filenames_list)

        Raises:
            ValueError: If no valid embeddings are found
        """
        embeddings = []
        filenames = []

        # Find all JSON files
        json_files = list(embeddings_dir.rglob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {embeddings_dir}")
            raise ValueError(f"No JSON files found in {embeddings_dir}")

        logger.info(f"Loading embeddings from {len(json_files)} files...")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'embedding_vector' in data and isinstance(data['embedding_vector'], list):
                    embeddings.append(data['embedding_vector'])
                    # Use relative filename without extension
                    relative_filename = json_file.relative_to(embeddings_dir)
                    filenames.append(str(relative_filename.with_suffix('')))
                else:
                    logger.warning(f"Invalid embedding format in {json_file.name}")

            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {e}")

        if not embeddings:
            raise ValueError("No valid embeddings found in directory")

        logger.info(f"✅ Successfully loaded {len(embeddings)} embeddings")
        return np.array(embeddings), filenames

    @staticmethod
    def save_results(results: Dict[str, Any], output_dir: Path) -> None:
        """
        Save clustering results to organized multi-file structure.

        Args:
            results: Dictionary containing clustering results
            output_dir: Directory where result files should be saved

        Raises:
            IOError: If files cannot be written
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save cluster assignments in separate directory
            IOHandler._save_cluster_assignments(results, output_dir)

            # Save quality analysis
            IOHandler._save_quality_report(results, output_dir)

            # Save mathematical analysis
            IOHandler._save_mathematical_analysis(results, output_dir)

            # Save metadata
            IOHandler._save_metadata(results, output_dir)

            logger.info(f"✅ Clustering results saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise IOError(f"Could not save results to {output_dir}: {e}")

    @staticmethod
    def _save_cluster_assignments(results: Dict[str, Any], output_dir: Path) -> None:
        """Save cluster assignments for each level."""
        assignments_dir = output_dir / "cluster_assignments"
        assignments_dir.mkdir(exist_ok=True)

        optimal_levels = results.get('optimal_levels', {})

        # Define ordering for level names (coarse to granular)
        level_order = {
            'primary': 1,
            'coarse': 1,
            'medium': 2,
            'fine': 3,
            'granular': 4,
            'detailed': 5
        }

        for level_name, level_data in optimal_levels.items():
            if level_data:
                # Simple structure with just essential info
                cluster_assignments = {
                    'level_name': level_name,
                    'n_clusters': level_data['n_clusters'],
                    'clusters': level_data['clusters']
                }

                # Add number prefix for proper sorting
                level_num = level_order.get(level_name, 0)
                assignment_file = assignments_dir / f"{level_num:02d}_{level_name}.json"
                with open(assignment_file, 'w', encoding='utf-8') as f:
                    json.dump(cluster_assignments, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_quality_report(results: Dict[str, Any], output_dir: Path) -> None:
        """Save quality analysis and level comparisons."""
        optimal_levels = results.get('optimal_levels', {})

        quality_report = {
            'level_comparison': {},
            'best_level': None,
            'summary': results.get('summary', {}),
            'recommendations': results.get('recommendations', {})
        }

        # Compare all levels
        best_quality = 0
        best_level = None

        for level_name, level_data in optimal_levels.items():
            if level_data:
                quality_score = level_data['quality_score']
                quality_report['level_comparison'][level_name] = {
                    'n_clusters': level_data['n_clusters'],
                    'quality_score': quality_score,
                    'silhouette': level_data['silhouette'],
                    'balance_score': level_data['balance_score'],
                    'separation_ratio': level_data['separation_ratio']
                }

                if quality_score > best_quality:
                    best_quality = quality_score
                    best_level = level_name

        quality_report['best_level'] = {
            'level': best_level,
            'quality_score': best_quality,
            'reason': 'Highest mathematical quality score'
        }

        report_file = output_dir / "quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_mathematical_analysis(results: Dict[str, Any], output_dir: Path) -> None:
        """Save complete mathematical analysis data."""
        import datetime

        # Simple structure with just the mathematical data
        mathematical_data = {
            'all_evaluations': results.get('all_evaluations', {}),
            'linkage_matrix': results.get('linkage_matrix', [])
        }

        # Convert numpy arrays in linkage matrix
        if mathematical_data['linkage_matrix'] is not None:
            mathematical_data['linkage_matrix'] = IOHandler._convert_numpy_types(
                mathematical_data['linkage_matrix']
            )

        analysis_file = output_dir / "mathematical_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(mathematical_data, f, indent=2, ensure_ascii=False,
                     default=IOHandler._convert_numpy_types)

    @staticmethod
    def _save_metadata(results: Dict[str, Any], output_dir: Path) -> None:
        """Save run metadata and configuration."""
        import datetime

        # Get complete configuration details
        config_used = results.get('summary', {}).get('config_used', {})

        metadata = {
            'generated_at': datetime.datetime.now().isoformat(),
            'total_embeddings': results.get('summary', {}).get('total_embeddings', 0),
            'total_heights_evaluated': results.get('summary', {}).get('total_heights_evaluated', 0),
            'natural_levels_found': results.get('summary', {}).get('natural_levels_found', 0),
            'discovery_method': results.get('summary', {}).get('discovery_method', 'unknown'),
            'config_used': {
                'min_clusters': config_used.get('min_clusters', 3),
                'target_count': config_used.get('target_count', 200),
                'distance_metric': config_used.get('distance_metric', 'cosine'),
                'linkage_method': config_used.get('linkage_method', 'ward'),
                'quality_weights': config_used.get('quality_weights', {}),
                'peak_detection': config_used.get('peak_detection', {})
            },
            'levels_discovered': list(results.get('optimal_levels', {}).keys()),
            'filenames': results.get('filenames', [])
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object

        Raises:
            TypeError: If object type is not supported
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")