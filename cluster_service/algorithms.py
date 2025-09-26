"""
Pure mathematical algorithms for hierarchical clustering.

This module contains all the mathematical functions for clustering analysis,
with no side effects or I/O operations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances


logger = logging.getLogger(__name__)


def build_linkage_matrix(embeddings: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    Build hierarchical clustering linkage matrix.

    Args:
        embeddings: Array of embedding vectors
        method: Linkage method ('ward', 'complete', 'average', etc.)

    Returns:
        Hierarchical clustering linkage matrix
    """
    logger.info("Building hierarchical linkage matrix...")
    distance_matrix = cosine_distances(embeddings)
    condensed_distances = squareform(distance_matrix)
    return linkage(condensed_distances, method=method)


def find_natural_breakpoints(linkage_matrix: np.ndarray,
                           min_clusters: int = 3,
                           target_count: int = 200) -> List[float]:
    """
    Find natural breakpoints with improved height distribution.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        min_clusters: Minimum number of clusters to consider
        target_count: Target number of heights to test

    Returns:
        List of heights at natural breakpoints
    """
    heights = linkage_matrix[:, 2]
    min_height, max_height = heights.min(), heights.max()

    # Method 1: Even distribution across meaningful clustering range
    clustering_min = min_height
    clustering_max = max_height * 0.85
    even_heights = np.linspace(clustering_min, clustering_max, 60)

    # Method 2: Target specific cluster count ranges
    target_cluster_counts = list(range(3, 121))
    target_heights = []

    for target_n in target_cluster_counts:
        low, high = clustering_min, clustering_max
        for _ in range(10):
            mid = (low + high) / 2
            n_clusters = len(set(fcluster(linkage_matrix, mid, criterion='distance')))
            if n_clusters > target_n:
                low = mid
            else:
                high = mid
        target_heights.append(mid)

    # Method 3: Include actual merge heights
    merge_heights = np.sort(np.unique(heights))
    if len(merge_heights) > 60:
        indices = np.linspace(0, len(merge_heights) - 1, 60, dtype=int)
        merge_heights = merge_heights[indices]

    # Combine all methods and remove duplicates
    all_heights = np.unique(np.concatenate([even_heights, target_heights, merge_heights]))

    # Filter out heights that give too few clusters
    valid_candidates = []
    for height in all_heights:
        n_clusters = len(set(fcluster(linkage_matrix, height, criterion='distance')))
        if n_clusters >= min_clusters:
            valid_candidates.append(float(height))

    # Sample to target count if needed
    if len(valid_candidates) > target_count:
        indices = np.linspace(0, len(valid_candidates) - 1, target_count, dtype=int)
        valid_candidates = [valid_candidates[i] for i in indices]

    logger.info(f"Generated {len(valid_candidates)} height candidates (target: {target_count})")
    return sorted(valid_candidates)


def evaluate_cut_height(embeddings: np.ndarray,
                        linkage_matrix: np.ndarray,
                        height: float,
                        quality_weights: Dict[str, float]) -> Dict:
    """
    Evaluate clustering quality at a given cut height.

    Args:
        embeddings: Original embedding vectors
        linkage_matrix: Hierarchical clustering linkage matrix
        height: Cut height to evaluate
        quality_weights: Weights for composite score calculation

    Returns:
        Dictionary with validity, metrics, and quality scores
    """
    clusters = fcluster(linkage_matrix, height, criterion='distance')
    n_clusters = len(set(clusters))

    if n_clusters < 2:
        return {'valid': False, 'reason': 'insufficient_clusters'}

    metrics = {}

    # Silhouette Analysis
    try:
        silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
        silhouette_samples_scores = silhouette_samples(embeddings, clusters, metric='cosine')

        metrics['silhouette'] = {
            'average': float(silhouette_avg),
            'std': float(np.std(silhouette_samples_scores)),
            'min': float(np.min(silhouette_samples_scores)),
            'negative_ratio': float(np.sum(silhouette_samples_scores < 0) / len(silhouette_samples_scores))
        }
    except Exception as e:
        logger.warning(f"Silhouette calculation failed: {e}")
        metrics['silhouette'] = {'average': -1, 'std': 1, 'min': -1, 'negative_ratio': 1}

    # Cluster Balance
    cluster_sizes = [np.sum(clusters == i) for i in set(clusters)]
    size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes)

    metrics['balance'] = {
        'cv': float(size_cv),
        'min_ratio': float(min(cluster_sizes) / len(clusters)),
        'max_ratio': float(max(cluster_sizes) / len(clusters)),
        'balance_score': float(1 / (1 + size_cv))
    }

    # Intra-cluster vs Inter-cluster distances
    metrics['separation'] = _calculate_separation_metrics(embeddings, clusters)

    # Stability placeholder
    metrics['stability'] = {'score': 0.5}

    # Composite Quality Score
    metrics['composite'] = _calculate_composite_score(metrics, quality_weights)

    return {
        'valid': True,
        'height': float(height),
        'n_clusters': n_clusters,
        'labels': clusters,
        'metrics': metrics
    }


def detect_natural_breaks(evaluations: Dict,
                          embeddings: np.ndarray,
                          peak_params: Dict[str, Optional[float]]) -> List[Dict]:
    """
    Detect natural hierarchy breaks using signal processing.

    Args:
        evaluations: Dictionary of quality evaluations at different heights
        embeddings: Original embeddings for context
        peak_params: Peak detection parameters

    Returns:
        List of natural break points with their properties
    """
    # Convert evaluations to arrays for signal processing
    evals_by_clusters = sorted(evaluations.items(), key=lambda x: x[1]['n_clusters'])

    # Filter to reasonable clustering range
    reasonable_evals = [(height, eval_data) for height, eval_data in evals_by_clusters
                       if 3 <= eval_data['n_clusters'] <= 300]

    if len(reasonable_evals) < 5:
        logger.warning("Not enough reasonable evaluations for peak detection")
        return []

    # Extract metrics for analysis
    clusters = np.array([eval_data['n_clusters'] for _, eval_data in reasonable_evals])
    silhouettes = np.array([eval_data['metrics']['silhouette']['average']
                           for _, eval_data in reasonable_evals])
    qualities = np.array([eval_data['metrics']['composite']['score']
                         for _, eval_data in reasonable_evals])
    heights = np.array([float(height) for height, _ in reasonable_evals])

    natural_breaks = []

    # Find peaks in silhouette scores
    peaks, peak_properties = find_peaks(
        silhouettes,
        height=peak_params.get('height'),
        distance=int(peak_params.get('distance', 3)),
        prominence=peak_params.get('prominence', 0.005),
        width=peak_params.get('width', 1)
    )

    for i, peak_idx in enumerate(peaks):
        natural_breaks.append({
            'cluster_count': int(clusters[peak_idx]),
            'silhouette': float(silhouettes[peak_idx]),
            'quality': float(qualities[peak_idx]),
            'height': float(heights[peak_idx]),
            'evaluation': reasonable_evals[peak_idx][1],
            'prominence': float(peak_properties['prominences'][i]),
            'break_type': 'peak',
            'signal': 'silhouette'
        })

    # Find valleys (local minima)
    valleys, valley_properties = find_peaks(
        -silhouettes,
        distance=int(peak_params.get('distance', 3)),
        prominence=peak_params.get('prominence', 0.005),
        width=peak_params.get('width', 1)
    )

    for i, valley_idx in enumerate(valleys):
        # Check if valley is a transition point
        is_transition = (0 < valley_idx < len(silhouettes) - 1 and
                        silhouettes[valley_idx-1] > silhouettes[valley_idx] and
                        silhouettes[valley_idx+1] > silhouettes[valley_idx])

        if is_transition or valley_properties['prominences'][i] > 0.01:
            natural_breaks.append({
                'cluster_count': int(clusters[valley_idx]),
                'silhouette': float(silhouettes[valley_idx]),
                'quality': float(qualities[valley_idx]),
                'height': float(heights[valley_idx]),
                'evaluation': reasonable_evals[valley_idx][1],
                'prominence': float(valley_properties['prominences'][i]),
                'break_type': 'valley',
                'signal': 'silhouette'
            })

    # Find gradient changes
    quality_gradient = np.gradient(qualities)
    gradient_peaks, grad_properties = find_peaks(
        np.abs(quality_gradient),
        prominence=np.std(quality_gradient) * 0.5,
        distance=int(peak_params.get('distance', 3))
    )

    for i, grad_idx in enumerate(gradient_peaks):
        if grad_idx not in peaks and grad_idx not in valleys:
            natural_breaks.append({
                'cluster_count': int(clusters[grad_idx]),
                'silhouette': float(silhouettes[grad_idx]),
                'quality': float(qualities[grad_idx]),
                'height': float(heights[grad_idx]),
                'evaluation': reasonable_evals[grad_idx][1],
                'prominence': float(grad_properties['prominences'][i]),
                'break_type': 'gradient',
                'signal': 'quality'
            })

    # Remove duplicates (within 5% cluster count)
    unique_breaks = []
    for nb in natural_breaks:
        is_duplicate = False
        for ub in unique_breaks:
            ratio = abs(nb['cluster_count'] - ub['cluster_count']) / max(nb['cluster_count'], ub['cluster_count'])
            if ratio < 0.05:
                if nb['quality'] > ub['quality']:
                    unique_breaks.remove(ub)
                    unique_breaks.append(nb)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_breaks.append(nb)

    return sorted(unique_breaks, key=lambda x: x['cluster_count'])


def select_hierarchy_levels(natural_breaks: List[Dict], max_levels: int = 5) -> List[Dict]:
    """
    Select the most appropriate natural breaks for hierarchy levels.

    Args:
        natural_breaks: List of detected natural breaks
        max_levels: Maximum number of hierarchy levels to select

    Returns:
        Selected breaks for hierarchy levels
    """
    if len(natural_breaks) <= max_levels:
        return natural_breaks

    # Sort by prominence to get the most significant peaks
    breaks_by_prominence = sorted(natural_breaks, key=lambda x: x['prominence'], reverse=True)

    # Select peaks that are well-separated in cluster count
    selected = []
    for break_info in breaks_by_prominence:
        cluster_count = break_info['cluster_count']

        # Check if this break is well-separated from already selected ones
        well_separated = True
        for selected_break in selected:
            separation = abs(cluster_count - selected_break['cluster_count'])
            min_separation = max(10, min(cluster_count, selected_break['cluster_count']) * 0.3)

            if separation < min_separation:
                well_separated = False
                break

        if well_separated:
            selected.append(break_info)

        if len(selected) >= max_levels:
            break

    # Sort selected breaks by cluster count for proper hierarchy order
    return sorted(selected, key=lambda x: x['cluster_count'])


# Private helper functions

def _calculate_separation_metrics(embeddings: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    """Calculate intra-cluster vs inter-cluster distance metrics."""
    intra_distances = []
    inter_distances = []

    for cluster_id in set(clusters):
        cluster_mask = clusters == cluster_id
        cluster_points = embeddings[cluster_mask]

        if len(cluster_points) > 1:
            # Sample distances for efficiency
            n_samples = min(len(cluster_points), 50)
            sampled_indices = np.random.choice(len(cluster_points), n_samples, replace=False)
            for i in sampled_indices[:10]:
                for j in sampled_indices[i+1:min(i+11, len(sampled_indices))]:
                    intra_distances.append(np.linalg.norm(cluster_points[i] - cluster_points[j]))

        # Inter-cluster distances
        other_points = embeddings[~cluster_mask]
        if len(other_points) > 0 and len(cluster_points) > 0:
            n_samples = min(10, len(cluster_points), len(other_points))
            for _ in range(n_samples):
                cluster_idx = np.random.randint(len(cluster_points))
                other_idx = np.random.randint(len(other_points))
                inter_distances.append(np.linalg.norm(cluster_points[cluster_idx] - other_points[other_idx]))

    if intra_distances and inter_distances:
        separation_ratio = np.mean(inter_distances) / (np.mean(intra_distances) + 1e-10)
        return {
            'ratio': float(separation_ratio),
            'intra_mean': float(np.mean(intra_distances)),
            'inter_mean': float(np.mean(inter_distances))
        }
    else:
        return {'ratio': 0.0, 'intra_mean': 0.0, 'inter_mean': 0.0}


def _calculate_composite_score(metrics: Dict, weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate weighted composite quality score."""
    # Normalize scores to 0-1 range
    sil_score = max(0, (metrics['silhouette']['average'] + 1) / 2)
    balance_score = metrics['balance']['balance_score']
    sep_score = min(1, metrics['separation']['ratio'] / 3)
    stab_score = metrics['stability']['score']

    composite = (
        weights['silhouette'] * sil_score +
        weights['balance'] * balance_score +
        weights['separation'] * sep_score +
        weights['stability'] * stab_score
    )

    return {
        'score': float(composite),
        'components': {
            'silhouette': float(sil_score),
            'balance': float(balance_score),
            'separation': float(sep_score),
            'stability': float(stab_score)
        }
    }