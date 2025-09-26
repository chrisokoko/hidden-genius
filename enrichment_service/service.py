"""
Main enrichment service that enriches clustering data with essence and Claude naming.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import EnrichmentConfig


logger = logging.getLogger(__name__)


class EnrichmentService:
    """
    Enrichment service that creates flat cluster structure with essence data.
    """

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """Initialize the enrichment service."""
        self._config = config or EnrichmentConfig()

    def load_cluster_assignments(self, clusters_dir: Path) -> Dict[str, Dict]:
        """Load cluster assignments from organized cluster files."""
        cluster_assignments = {}
        assignments_dir = clusters_dir / "cluster_assignments"

        if not assignments_dir.exists():
            raise FileNotFoundError(f"Cluster assignments directory not found: {assignments_dir}")

        for cluster_file in sorted(assignments_dir.glob("*.json")):
            try:
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                level_name = data.get('level_name')

                if level_name:
                    # Store the entire level data (includes metadata + clusters)
                    cluster_assignments[level_name] = data
                    clusters = data.get('clusters', {})
                    logger.info(f"Loaded {level_name} level: {len(clusters)} clusters")

            except Exception as e:
                logger.error(f"Error loading cluster file {cluster_file}: {e}")

        return cluster_assignments

    def load_fingerprint(self, filename: str) -> Optional[Dict]:
        """Load fingerprint JSON file for a given filename."""
        fingerprints_dir = Path(self._config.fingerprints_dir)
        fingerprint_path = fingerprints_dir / f"{filename}.json"

        if not fingerprint_path.exists():
            logger.warning(f"Fingerprint not found: {fingerprint_path}")
            return None

        try:
            with open(fingerprint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading fingerprint {filename}: {e}")
            return None

    def create_flat_clusters(self, cluster_assignments: Dict[str, Dict]) -> List[Dict]:
        """Create flat cluster structure with essence data."""
        flat_clusters = []

        for level_name, level_data in cluster_assignments.items():
            clusters = level_data.get('clusters', {})
            logger.info(f"Processing {level_name} level with {len(clusters)} clusters")

            for cluster_id, filenames in clusters.items():
                # Create flat cluster object (remove redundant "level" field)
                flat_cluster = {
                    "cluster_id": f"{level_name}_{cluster_id}",
                    "files": []
                }

                # Load fingerprints for each file
                for filename in filenames:
                    fingerprint = self.load_fingerprint(filename)

                    file_data = {
                        "filename": filename,
                        "title": None,
                        "essence": None
                    }

                    if fingerprint:
                        file_data["title"] = fingerprint.get("title")
                        file_data["essence"] = fingerprint.get("essence")

                    flat_cluster["files"].append(file_data)

                flat_clusters.append(flat_cluster)

        logger.info(f"Created {len(flat_clusters)} flat clusters")
        return flat_clusters


    def save_enriched_clusters_by_level(self, flat_clusters: List[Dict],
                                        cluster_assignments: Dict[str, Dict], output_dir: Path) -> None:
        """Save enriched clusters in the same structure as original cluster assignments."""
        essences_dir = output_dir
        essences_dir.mkdir(parents=True, exist_ok=True)

        # Group clusters by level
        by_level = {}
        for cluster in flat_clusters:
            # Extract level from cluster_id (e.g., "coarse_4" -> "coarse")
            level = cluster["cluster_id"].split("_")[0]
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(cluster)

        # Level ordering for numbered filenames
        level_order = {
            'coarse': 1,
            'medium': 2,
            'fine': 3,
            'granular': 4,
            'detailed': 5
        }

        # Save each level to its own file
        for level_name, clusters in by_level.items():
            level_num = level_order.get(level_name, 0)
            filename = f"{level_num:02d}_{level_name}.json"
            level_file = essences_dir / filename

            # Get original cluster assignment data for this level
            original_level_data = cluster_assignments.get(level_name, {})

            # Structure for this level - include all original metadata
            level_data = {
                "level_name": level_name,
                "n_clusters": original_level_data.get("n_clusters", len(clusters))
            }

            # Add size statistics (either distribution or individual sizes)
            if "cluster_size_distribution" in original_level_data:
                level_data["cluster_size_distribution"] = original_level_data["cluster_size_distribution"]
            elif "individual_cluster_sizes" in original_level_data:
                level_data["individual_cluster_sizes"] = original_level_data["individual_cluster_sizes"]

            # Add the enriched clusters
            level_data["clusters"] = clusters

            with open(level_file, 'w', encoding='utf-8') as f:
                json.dump(level_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Saved {level_name} level: {level_file}")

    def run(self, clusters_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Run the complete enrichment pipeline."""
        logger.info("🌟 Starting Cluster Enrichment Service")
        logger.info("=" * 60)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Load cluster assignments
            cluster_assignments = self.load_cluster_assignments(clusters_dir)

            if not cluster_assignments:
                logger.warning("No cluster assignments found")
                return {'processed': 0, 'failed': 0, 'skipped': 0}

            # Step 2: Create flat structure with essence data
            flat_clusters = self.create_flat_clusters(cluster_assignments)

            # Step 3: Skip Claude naming for now - just use flat clusters with essences
            enriched_clusters = flat_clusters

            # Step 4: Save enriched clusters by level in organized structure
            self.save_enriched_clusters_by_level(enriched_clusters, cluster_assignments, output_dir)

            # Log summary
            self._log_enrichment_summary(enriched_clusters)

            return {'processed': len(enriched_clusters), 'failed': 0, 'skipped': 0}

        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            return {'processed': 0, 'failed': 1, 'skipped': 0}

    def _log_enrichment_summary(self, enriched_clusters: List[Dict]) -> None:
        """Log summary of enrichment results."""
        logger.info("\\n📋 ENRICHMENT SUMMARY:")

        # Group by level (extract from cluster_id)
        by_level = {}
        for cluster in enriched_clusters:
            level = cluster["cluster_id"].split("_")[0]
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(cluster)

        for level_name, clusters in by_level.items():
            total_clusters = len(clusters)
            clusters_with_essence = sum(1 for cluster in clusters
                                      if any(f.get('essence') for f in cluster['files']))

            logger.info(f"🎯 {level_name.title()} Level:")
            logger.info(f"   Total Clusters: {total_clusters}")
            logger.info(f"   Clusters with Essence: {clusters_with_essence}")
            if total_clusters > 0:
                logger.info(f"   Coverage: {clusters_with_essence/total_clusters*100:.1f}%")