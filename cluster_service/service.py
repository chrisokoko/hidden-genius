"""
Clustering service with clean architecture and separation of concerns.

This module provides the main ClusteringService class that orchestrates
the clustering pipeline using dependency injection and clean interfaces.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .algorithms import (
    build_linkage_matrix,
    detect_natural_breaks,
    evaluate_cut_height,
    find_natural_breakpoints,
    select_hierarchy_levels
)
from .config import ClusteringConfig
from .io_handler import IOHandler


logger = logging.getLogger(__name__)


class ClusteringService:
    """
    Main clustering service that orchestrates the clustering pipeline.

    Uses dependency injection for configuration and clean separation of concerns
    between I/O operations, mathematical algorithms, and business logic.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the clustering service.

        Args:
            config: Clustering configuration. If None, loads default config.
        """
        self._config = config or ClusteringConfig()
        self._io_handler = IOHandler()
        self._logger = logger

    def run(self, embeddings_dir: Path, output_dir: Path) -> Dict[str, int]:
        """
        Run the complete clustering pipeline.

        Args:
            embeddings_dir: Directory containing embedding JSON files
            output_dir: Directory where results will be saved

        Returns:
            Dictionary with processing statistics

        Raises:
            ValueError: If input directory is invalid
            IOError: If results cannot be saved
        """
        self._logger.info("🎯 Starting Intelligent Hierarchical Clustering")
        self._logger.info("=" * 60)

        # Validate input
        self._validate_input_directory(embeddings_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load embeddings
            embeddings, filenames = self._io_handler.load_embeddings(embeddings_dir)

            if len(embeddings) == 0:
                self._logger.warning("No embeddings found in directory")
                return {'processed': 0, 'failed': 0, 'skipped': 0}

            # Perform clustering
            results = self._perform_clustering(embeddings, filenames)

            # Save results to organized multi-file structure
            self._io_handler.save_results(results, output_dir)

            # Log summary
            self._log_clustering_summary(results)

            return {'processed': 1, 'failed': 0, 'skipped': 0}

        except Exception as e:
            self._logger.error(f"Clustering failed: {e}")
            return {'processed': 0, 'failed': 1, 'skipped': 0}

    def _validate_input_directory(self, embeddings_dir: Path) -> None:
        """Validate that input directory exists and is accessible."""
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory does not exist: {embeddings_dir}")

        if not embeddings_dir.is_dir():
            raise ValueError(f"Path is not a directory: {embeddings_dir}")

    def _perform_clustering(self, embeddings: np.ndarray, filenames: List[str]) -> Dict:
        """
        Perform the hierarchical clustering analysis.

        Args:
            embeddings: Array of embedding vectors
            filenames: List of filenames corresponding to embeddings

        Returns:
            Complete clustering results dictionary
        """
        num_points = len(embeddings)
        self._logger.info(f"Starting clustering on {num_points} embeddings")

        # Handle edge cases
        if num_points == 0:
            return self._create_empty_result()
        if num_points == 1:
            return self._create_single_point_result(filenames)

        # Build linkage matrix
        linkage_matrix = build_linkage_matrix(embeddings, self._config.linkage_method)

        # Find candidate breakpoints
        self._logger.info("Phase 1: Detecting natural breakpoints...")
        breakpoint_heights = find_natural_breakpoints(
            linkage_matrix, self._config.min_clusters, self._config.target_count
        )
        self._logger.info(f"Found {len(breakpoint_heights)} natural breakpoint candidates")

        # Evaluate quality at each breakpoint
        self._logger.info("Phase 2: Evaluating quality at each breakpoint...")
        evaluations = self._evaluate_all_breakpoints(
            embeddings, linkage_matrix, breakpoint_heights
        )

        # Detect natural hierarchy breaks
        self._logger.info("Phase 3: Detecting natural hierarchy breaks...")
        optimal_levels = self._find_optimal_levels(evaluations, embeddings)

        # Format and return results
        return self._format_results(
            linkage_matrix, evaluations, optimal_levels, filenames, embeddings
        )

    def _evaluate_all_breakpoints(self, embeddings: np.ndarray,
                                  linkage_matrix: np.ndarray,
                                  heights: List[float]) -> Dict:
        """Evaluate clustering quality at all candidate heights."""
        evaluations = {}

        for height in heights:
            eval_result = evaluate_cut_height(
                embeddings, linkage_matrix, height, self._config.quality_weights
            )

            if eval_result['valid']:
                evaluations[height] = eval_result
                self._logger.info(
                    f"  Height {height:.3f}: {eval_result['n_clusters']} clusters, "
                    f"quality={eval_result['metrics']['composite']['score']:.3f}"
                )

        return evaluations

    def _find_optimal_levels(self, evaluations: Dict, embeddings: np.ndarray) -> Dict:
        """Find optimal clustering levels using natural break detection."""
        natural_breaks = detect_natural_breaks(
            evaluations, embeddings, self._config.peak_detection
        )

        if not natural_breaks:
            self._logger.warning("No natural breaks found in the data")
            return {}

        self._logger.info(f"Found {len(natural_breaks)} unique natural breaks")

        # Log discovered breaks
        for nb in natural_breaks:
            self._logger.info(
                f"{nb['break_type'].title()}: {nb['cluster_count']} clusters, "
                f"sil={nb['silhouette']:.3f}, qual={nb['quality']:.3f}, "
                f"prom={nb['prominence']:.3f}"
            )

        # Select hierarchy levels
        selected_breaks = select_hierarchy_levels(natural_breaks, max_levels=5)

        # Convert to expected format
        return self._convert_breaks_to_levels(selected_breaks)

    def _convert_breaks_to_levels(self, selected_breaks: List[Dict]) -> Dict:
        """Convert selected breaks to hierarchical level format."""
        optimal_levels = {}

        # Dynamic level naming based on number of breaks found
        level_names = self._get_level_names(len(selected_breaks))

        for i, break_info in enumerate(selected_breaks):
            if i < len(level_names):
                level_name = level_names[i]
                optimal_levels[level_name] = (break_info['height'], break_info['evaluation'])

                self._logger.info(
                    f"Natural {level_name} level: {break_info['cluster_count']} clusters, "
                    f"silhouette={break_info['silhouette']:.3f}, "
                    f"quality={break_info['quality']:.3f}, type={break_info['break_type']}"
                )

        return optimal_levels

    def _get_level_names(self, num_levels: int) -> List[str]:
        """Get appropriate level names based on number of levels found."""
        if num_levels == 1:
            return ['primary']
        elif num_levels == 2:
            return ['coarse', 'fine']
        elif num_levels == 3:
            return ['coarse', 'medium', 'fine']
        elif num_levels == 4:
            return ['coarse', 'medium', 'fine', 'granular']
        else:  # 5 or more
            return ['coarse', 'medium', 'fine', 'granular', 'detailed']

    def _format_results(self, linkage_matrix: np.ndarray, evaluations: Dict,
                       optimal_levels: Dict, filenames: List[str],
                       embeddings: np.ndarray) -> Dict:
        """Format complete clustering results."""
        results = {
            'linkage_matrix': linkage_matrix,
            'all_evaluations': evaluations,
            'optimal_levels': {},
            'recommendations': {},
            'filenames': filenames
        }

        # Format each level
        for level, level_data in optimal_levels.items():
            if level_data is not None:
                height, eval_data = level_data
                clusters = self._create_cluster_mapping(eval_data['labels'], filenames)

                results['optimal_levels'][level] = {
                    'height': float(height),
                    'n_clusters': int(eval_data['n_clusters']),
                    'quality_score': float(eval_data['metrics']['composite']['score']),
                    'silhouette': float(eval_data['metrics']['silhouette']['average']),
                    'balance_score': float(eval_data['metrics']['balance']['balance_score']),
                    'separation_ratio': float(eval_data['metrics']['separation']['ratio']),
                    'clusters': clusters,
                    'labels': eval_data['labels'].tolist()
                }

                results['recommendations'][level] = {
                    'height': float(height),
                    'n_clusters': int(eval_data['n_clusters']),
                    'quality_score': float(eval_data['metrics']['composite']['score']),
                    'silhouette': float(eval_data['metrics']['silhouette']['average']),
                    'labels': eval_data['labels'].tolist(),
                    'description': f"{eval_data['n_clusters']} {level} level clusters (natural break detected)"
                }

        # Add summary statistics
        results['summary'] = self._create_summary(embeddings, evaluations, optimal_levels)

        return results

    def _create_cluster_mapping(self, labels: np.ndarray, filenames: List[str]) -> Dict[str, List[str]]:
        """Create mapping of cluster IDs to filenames."""
        clusters = {}
        for idx, cluster_id in enumerate(labels):
            cluster_key = str(int(cluster_id))
            if cluster_key not in clusters:
                clusters[cluster_key] = []

            filename = filenames[idx] if idx < len(filenames) else f"item_{idx}"
            clusters[cluster_key].append(filename)

        return clusters

    def _create_summary(self, embeddings: np.ndarray, evaluations: Dict,
                       optimal_levels: Dict) -> Dict:
        """Create summary statistics."""
        return {
            'total_embeddings': len(embeddings),
            'total_heights_evaluated': len(evaluations),
            'natural_levels_found': len([l for l in optimal_levels.values() if l is not None]),
            'discovery_method': 'mathematical_peak_detection',
            'config_used': {
                'min_clusters': self._config.min_clusters,
                'target_count': self._config.target_count,
                'distance_metric': self._config.distance_metric,
                'linkage_method': self._config.linkage_method,
                'quality_weights': self._config.quality_weights,
                'peak_detection': self._config.peak_detection
            }
        }

    def _log_clustering_summary(self, results: Dict) -> None:
        """Log summary of clustering results."""
        self._logger.info("\n📋 NATURAL HIERARCHY DISCOVERY:")

        for level, rec in results['recommendations'].items():
            if rec:
                self._logger.info(f"🎯 {level.title()} Level:")
                self._logger.info(f"   Cut Height: {rec['height']:.3f}")
                self._logger.info(f"   Clusters: {rec['n_clusters']}")
                self._logger.info(f"   Quality: {rec['quality_score']:.3f}")
                self._logger.info(f"   Silhouette: {rec['silhouette']:.3f}")

        if results['optimal_levels']:
            for level, data in results['optimal_levels'].items():
                self._logger.info(
                    f"🎯 {level.title()} Level: {data['n_clusters']} clusters, "
                    f"quality={data['quality_score']:.3f}"
                )

    def _create_empty_result(self) -> Dict:
        """Create result for when there are no embeddings."""
        return {
            'optimal_levels': {},
            'all_evaluations': {},
            'recommendations': {},
            'summary': {
                'total_embeddings': 0,
                'total_heights_evaluated': 0,
                'natural_levels_found': 0,
                'discovery_method': 'empty_result'
            },
            'filenames': []
        }

    def _create_single_point_result(self, filenames: List[str]) -> Dict:
        """Create result for when there's only one embedding."""
        return {
            'optimal_levels': {
                'single': {
                    'height': 0.0,
                    'n_clusters': 1,
                    'quality_score': 1.0,
                    'silhouette': 0.0,
                    'balance_score': 1.0,
                    'separation_ratio': 0.0,
                    'clusters': {'0': [filenames[0] if filenames else 'item_0']},
                    'labels': [0]
                }
            },
            'all_evaluations': {},
            'recommendations': {
                'single': {
                    'height': 0.0,
                    'n_clusters': 1,
                    'quality_score': 1.0,
                    'silhouette': 0.0,
                    'labels': [0],
                    'description': 'Single cluster (only one embedding provided)'
                }
            },
            'summary': {
                'total_embeddings': 1,
                'total_heights_evaluated': 0,
                'natural_levels_found': 1,
                'discovery_method': 'single_point'
            },
            'filenames': filenames
        }


