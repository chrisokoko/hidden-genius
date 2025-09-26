#!/usr/bin/env python3
"""
Export Semantic Clusters

Exports each coarse cluster as a separate JSON file containing:
- Coarse cluster metadata (title, description)
- All medium clusters within it (with titles, descriptions)
- All fine clusters within those (with titles, descriptions)
- All voice memos with full transcripts and fingerprints

Each coarse cluster becomes one analyzable file for Claude to extract "genius" insights.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clustering_results(file_path: str = "data/03 clusters/cluster_assignments.json") -> Optional[Dict[str, Any]]:
    """Load clustering results with hierarchy assignments"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Clustering results not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in clustering results: {e}")
        return None

def load_cluster_metadata(cluster_names_dir: str = "data/cluster_names") -> Optional[Dict[str, Any]]:
    """Load cluster titles and descriptions from batch files in cluster_names directory"""
    try:
        cluster_names_path = Path(cluster_names_dir)
        if not cluster_names_path.exists():
            logger.warning(f"Cluster names directory not found: {cluster_names_dir}")
            return None
        
        # Find all batch files
        batch_files = list(cluster_names_path.glob("*_batch_*.json"))
        if not batch_files:
            # Try single metadata file as fallback
            single_file = Path("data/clusters/cluster_metadata.json")
            if single_file.exists():
                logger.info("Using single cluster_metadata.json file")
                with open(single_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"No batch files found in: {cluster_names_dir}")
                return None
        
        logger.info(f"Loading cluster metadata from {len(batch_files)} batch files...")
        
        # Combine all cluster metadata
        combined_metadata = {
            "cluster_titles_descriptions": {
                "coarse": {},
                "medium": {},
                "fine": {}
            }
        }
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                # Extract cluster_titles_descriptions from this batch
                if 'cluster_titles_descriptions' in batch_data:
                    titles_descriptions = batch_data['cluster_titles_descriptions']
                    
                    # Merge each level
                    for level in ['coarse', 'medium', 'fine']:
                        if level in titles_descriptions:
                            combined_metadata['cluster_titles_descriptions'][level].update(
                                titles_descriptions[level]
                            )
                            
                logger.info(f"Loaded: {batch_file.name}")
            except Exception as e:
                logger.warning(f"Error loading {batch_file.name}: {e}")
        
        return combined_metadata
        
    except Exception as e:
        logger.error(f"Error loading cluster metadata: {e}")
        return None

def load_transcript(file_path: str, transcripts_base: str = "data/00 transcripts") -> Optional[str]:
    """Load transcript text from file"""
    try:
        full_path = Path(transcripts_base) / file_path.replace('.json', '.txt')
        if not full_path.exists():
            # Try without .txt extension
            full_path = Path(transcripts_base) / file_path.replace('.json', '')
            if not full_path.exists():
                logger.warning(f"Transcript not found: {full_path}")
                return None
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Error loading transcript {file_path}: {e}")
        return None

def load_fingerprint(file_path: str, fingerprints_base: str = "data/01 fingerprints") -> Optional[Dict[str, Any]]:
    """Load semantic fingerprint JSON"""
    try:
        # Ensure .json extension
        clean_path = file_path if file_path.endswith('.json') else file_path + '.json'
        full_path = Path(fingerprints_base) / clean_path
        
        if not full_path.exists():
            logger.warning(f"Fingerprint not found: {full_path}")
            return None
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading fingerprint {file_path}: {e}")
        return None

def extract_quality_metrics(fingerprint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract quality metrics from fingerprint data"""
    if not fingerprint:
        return {
            "raw_essence": None,
            "fingerprint_quality": None,
            "quality_coefficient": None
        }
    
    return {
        "raw_essence": fingerprint.get('raw_essence'),
        "fingerprint_quality": fingerprint.get('fingerprint_quality'),
        "quality_coefficient": fingerprint.get('insight_quality', {}).get('quality_coefficient')
    }


def sanitize_filename(title: str) -> str:
    """Convert cluster title to safe filename"""
    # Convert to lowercase, replace spaces and special chars with underscores
    safe_name = re.sub(r'[^\w\s-]', '', title.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    # Remove leading/trailing underscores and limit length
    return safe_name.strip('_')[:50]

def build_hierarchical_mapping(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build mapping from files to their cluster assignments at each level"""
    file_mapping = clustering_results.get('file_mapping', [])
    optimal_levels = clustering_results.get('optimal_levels', {})
    
    # Create mapping from file to index
    file_to_index = {file_path: idx for idx, file_path in enumerate(file_mapping)}
    
    # Get cluster labels for each level
    coarse_labels = optimal_levels.get('coarse', {}).get('labels', [])
    medium_labels = optimal_levels.get('medium', {}).get('labels', [])
    fine_labels = optimal_levels.get('fine', {}).get('labels', [])
    
    # Build reverse mapping: cluster_id -> list of files
    hierarchy = {
        'coarse': {},
        'medium': {},
        'fine': {}
    }
    
    # Map each file to its clusters
    file_clusters = {}
    for file_path, idx in file_to_index.items():
        file_clusters[file_path] = {
            'coarse': str(coarse_labels[idx]) if idx < len(coarse_labels) else None,
            'medium': str(medium_labels[idx]) if idx < len(medium_labels) else None,
            'fine': str(fine_labels[idx]) if idx < len(fine_labels) else None
        }
        
        # Add to hierarchy
        coarse_id = str(coarse_labels[idx]) if idx < len(coarse_labels) else None
        medium_id = str(medium_labels[idx]) if idx < len(medium_labels) else None
        fine_id = str(fine_labels[idx]) if idx < len(fine_labels) else None
        
        if coarse_id is not None:
            if coarse_id not in hierarchy['coarse']:
                hierarchy['coarse'][coarse_id] = []
            hierarchy['coarse'][coarse_id].append(file_path)
            
        if medium_id is not None:
            if medium_id not in hierarchy['medium']:
                hierarchy['medium'][medium_id] = []
            hierarchy['medium'][medium_id].append(file_path)
            
        if fine_id is not None:
            if fine_id not in hierarchy['fine']:
                hierarchy['fine'][fine_id] = []
            hierarchy['fine'][fine_id].append(file_path)
    
    return {
        'hierarchy': hierarchy,
        'file_clusters': file_clusters
    }

def export_coarse_cluster(coarse_id: str, 
                         coarse_files: List[str],
                         mapping_data: Dict[str, Any],
                         cluster_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Export a single coarse cluster with all its nested data"""
    
    hierarchy = mapping_data['hierarchy']
    file_clusters = mapping_data['file_clusters']
    
    # Get cluster metadata
    coarse_meta = {}
    medium_clusters_data = {}
    
    if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
        titles_desc = cluster_metadata['cluster_titles_descriptions']
        coarse_meta = titles_desc.get('coarse', {}).get(coarse_id, {})
    
    # Find all medium clusters that belong to this coarse cluster
    medium_clusters_in_coarse = set()
    for file_path in coarse_files:
        if file_path in file_clusters and file_clusters[file_path]['medium']:
            medium_clusters_in_coarse.add(file_clusters[file_path]['medium'])
    
    # Build medium clusters data
    for medium_id in medium_clusters_in_coarse:
        medium_files = hierarchy['medium'].get(medium_id, [])
        # Filter to only files that are also in this coarse cluster
        medium_files = [f for f in medium_files if f in coarse_files]
        
        if not medium_files:
            continue
            
        # Get medium cluster metadata
        medium_meta = {}
        if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
            titles_desc = cluster_metadata['cluster_titles_descriptions']
            medium_meta = titles_desc.get('medium', {}).get(medium_id, {})
        
        # Find fine clusters in this medium cluster
        fine_clusters_in_medium = set()
        for file_path in medium_files:
            if file_path in file_clusters and file_clusters[file_path]['fine']:
                fine_clusters_in_medium.add(file_clusters[file_path]['fine'])
        
        fine_clusters_data = {}
        for fine_id in fine_clusters_in_medium:
            fine_files = hierarchy['fine'].get(fine_id, [])
            # Filter to only files in this medium cluster
            fine_files = [f for f in fine_files if f in medium_files]
            
            if not fine_files:
                continue
                
            # Get fine cluster metadata  
            fine_meta = {}
            if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
                titles_desc = cluster_metadata['cluster_titles_descriptions']
                fine_meta = titles_desc.get('fine', {}).get(fine_id, {})
            
            # Load voice memos for this fine cluster
            voice_memos = []
            for file_path in fine_files:
                # Load transcript and fingerprint
                transcript = load_transcript(file_path)
                fingerprint = load_fingerprint(file_path)
                
                if transcript or fingerprint:
                    quality_metrics = extract_quality_metrics(fingerprint)
                    
                    memo_data = {
                        "filename": file_path,
                        "transcript": transcript,
                        "fingerprint": fingerprint,
                        **quality_metrics
                    }
                    
                    # Add title and description from fingerprint if available
                    if fingerprint:
                        memo_data["title"] = fingerprint.get('core_exploration', {}).get('central_question', 'Untitled')
                        memo_data["description"] = fingerprint.get('raw_essence', 'No description available')
                    else:
                        memo_data["title"] = file_path
                        memo_data["description"] = "Fingerprint not available"
                    
                    voice_memos.append(memo_data)
            
            fine_clusters_data[fine_id] = {
                "title": fine_meta.get('title', f'Fine Cluster {fine_id}'),
                "description": fine_meta.get('description', 'No description available'),
                "voice_memos": voice_memos
            }
        
        medium_clusters_data[medium_id] = {
            "title": medium_meta.get('title', f'Medium Cluster {medium_id}'),
            "description": medium_meta.get('description', 'No description available'), 
            "fine_clusters": fine_clusters_data
        }
    
    # Calculate statistics
    total_voice_memos = len(coarse_files)
    total_medium_clusters = len(medium_clusters_data)
    total_fine_clusters = sum(len(mc['fine_clusters']) for mc in medium_clusters_data.values())
    
    # Build final export structure
    export_data = {
        "cluster_metadata": {
            "cluster_id": coarse_id,
            "level": "coarse", 
            "title": coarse_meta.get('title', f'Coarse Cluster {coarse_id}'),
            "description": coarse_meta.get('description', 'No description available'),
            "export_type": "full",
            "exported_at": datetime.now().isoformat(),
            "statistics": {
                "total_voice_memos": total_voice_memos,
                "medium_clusters": total_medium_clusters,
                "fine_clusters": total_fine_clusters
            }
        },
        "medium_clusters": medium_clusters_data
    }
    
    return export_data

def export_condensed_clusters(coarse_id: str, 
                             coarse_files: List[str],
                             mapping_data: Dict[str, Any],
                             cluster_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Export a condensed version with only quality metrics"""
    
    hierarchy = mapping_data['hierarchy']
    file_clusters = mapping_data['file_clusters']
    
    # Get cluster metadata
    coarse_meta = {}
    if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
        titles_desc = cluster_metadata['cluster_titles_descriptions']
        coarse_meta = titles_desc.get('coarse', {}).get(coarse_id, {})
    
    # Find all medium clusters that belong to this coarse cluster
    medium_clusters_in_coarse = set()
    for file_path in coarse_files:
        if file_path in file_clusters and file_clusters[file_path]['medium']:
            medium_clusters_in_coarse.add(file_clusters[file_path]['medium'])
    
    # Build medium clusters data
    medium_clusters_data = {}
    for medium_id in medium_clusters_in_coarse:
        medium_files = hierarchy['medium'].get(medium_id, [])
        # Filter to only files that are also in this coarse cluster
        medium_files = [f for f in medium_files if f in coarse_files]
        
        if not medium_files:
            continue
            
        # Get medium cluster metadata
        medium_meta = {}
        if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
            titles_desc = cluster_metadata['cluster_titles_descriptions']
            medium_meta = titles_desc.get('medium', {}).get(medium_id, {})
        
        # Find fine clusters in this medium cluster
        fine_clusters_in_medium = set()
        for file_path in medium_files:
            if file_path in file_clusters and file_clusters[file_path]['fine']:
                fine_clusters_in_medium.add(file_clusters[file_path]['fine'])
        
        fine_clusters_data = {}
        for fine_id in fine_clusters_in_medium:
            fine_files = hierarchy['fine'].get(fine_id, [])
            # Filter to only files in this medium cluster
            fine_files = [f for f in fine_files if f in medium_files]
            
            if not fine_files:
                continue
                
            # Get fine cluster metadata  
            fine_meta = {}
            if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
                titles_desc = cluster_metadata['cluster_titles_descriptions']
                fine_meta = titles_desc.get('fine', {}).get(fine_id, {})
            
            # Load condensed voice memos for this fine cluster
            voice_memos = []
            for file_path in fine_files:
                # Load only fingerprint for quality metrics
                fingerprint = load_fingerprint(file_path)
                quality_metrics = extract_quality_metrics(fingerprint)
                
                memo_data = {
                    "filename": file_path,
                    **quality_metrics
                }
                
                voice_memos.append(memo_data)
            
            fine_clusters_data[fine_id] = {
                "title": fine_meta.get('title', f'Fine Cluster {fine_id}'),
                "description": fine_meta.get('description', 'No description available'),
                "voice_memos": voice_memos
            }
        
        medium_clusters_data[medium_id] = {
            "title": medium_meta.get('title', f'Medium Cluster {medium_id}'),
            "description": medium_meta.get('description', 'No description available'), 
            "fine_clusters": fine_clusters_data
        }
    
    # Calculate statistics
    total_voice_memos = len(coarse_files)
    total_medium_clusters = len(medium_clusters_data)
    total_fine_clusters = sum(len(mc['fine_clusters']) for mc in medium_clusters_data.values())
    
    # Build final export structure
    export_data = {
        "cluster_metadata": {
            "cluster_id": coarse_id,
            "level": "coarse", 
            "title": coarse_meta.get('title', f'Coarse Cluster {coarse_id}'),
            "description": coarse_meta.get('description', 'No description available'),
            "export_type": "condensed",
            "exported_at": datetime.now().isoformat(),
            "statistics": {
                "total_voice_memos": total_voice_memos,
                "medium_clusters": total_medium_clusters,
                "fine_clusters": total_fine_clusters
            }
        },
        "medium_clusters": medium_clusters_data
    }
    
    return export_data

def export_medium_clusters_condensed(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Export all medium clusters with condensed voice memo data"""
    
    # Get medium clusters from cluster assignments
    medium_clusters = clustering_results.get('levels', {}).get('medium', {}).get('clusters', {})
    
    if not medium_clusters:
        logger.error("No medium clusters found in clustering results")
        return {}
    
    clusters_data = {}
    total_voice_memos = 0
    
    for cluster_id, file_paths in medium_clusters.items():
        voice_memos = []
        
        for file_path in file_paths:
            # Load only fingerprint for quality metrics
            fingerprint = load_fingerprint(file_path)
            quality_metrics = extract_quality_metrics(fingerprint)
            
            memo_data = {
                "filename": file_path,
                **quality_metrics
            }
            
            voice_memos.append(memo_data)
            total_voice_memos += 1
        
        clusters_data[cluster_id] = {
            "voice_memos": voice_memos
        }
    
    # Build export structure
    export_data = {
        "export_metadata": {
            "export_type": "condensed",
            "level": "medium",
            "exported_at": datetime.now().isoformat(),
            "total_clusters": len(clusters_data),
            "total_voice_memos": total_voice_memos
        },
        "clusters": clusters_data
    }
    
    return export_data

def export_coarse_clusters_condensed(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Export all coarse clusters with condensed voice memo data"""
    
    # Get coarse clusters from cluster assignments
    coarse_clusters = clustering_results.get('levels', {}).get('coarse', {}).get('clusters', {})
    
    if not coarse_clusters:
        logger.error("No coarse clusters found in clustering results")
        return {}
    
    clusters_data = {}
    total_voice_memos = 0
    
    for cluster_id, file_paths in coarse_clusters.items():
        voice_memos = []
        
        for file_path in file_paths:
            # Load only fingerprint for quality metrics
            fingerprint = load_fingerprint(file_path)
            quality_metrics = extract_quality_metrics(fingerprint)
            
            memo_data = {
                "filename": file_path,
                **quality_metrics
            }
            
            voice_memos.append(memo_data)
            total_voice_memos += 1
        
        clusters_data[cluster_id] = {
            "voice_memos": voice_memos
        }
    
    # Build export structure
    export_data = {
        "export_metadata": {
            "export_type": "condensed",
            "level": "coarse",
            "exported_at": datetime.now().isoformat(),
            "total_clusters": len(clusters_data),
            "total_voice_memos": total_voice_memos
        },
        "clusters": clusters_data
    }
    
    return export_data

def calculate_average_quality_coefficient(voice_memos: List[Dict[str, Any]]) -> Optional[float]:
    """Calculate average quality coefficient for a list of voice memos"""
    quality_coefficients = []
    
    for memo in voice_memos:
        coeff = memo.get('quality_coefficient')
        if coeff is not None and isinstance(coeff, (int, float)):
            quality_coefficients.append(coeff)
    
    if not quality_coefficients:
        return None
    
    return sum(quality_coefficients) / len(quality_coefficients)

def filter_clusters_by_quality() -> int:
    """Filter enriched clusters by quality coefficient >= 0.6"""
    QUALITY_THRESHOLD = 0.6
    
    print("🔍 FILTERING ENRICHED CLUSTERS BY QUALITY")
    print("=" * 60)
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    
    # Create output directory
    output_dir = Path("data/05 filtered_enriched_clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and filter medium clusters
    medium_input_path = Path("data/04 enriched_clusters/medium_clusters_condensed.json")
    if not medium_input_path.exists():
        logger.error(f"Medium clusters file not found: {medium_input_path}")
        return 1
    
    print("📂 Loading medium clusters...")
    with open(medium_input_path, 'r', encoding='utf-8') as f:
        medium_data = json.load(f)
    
    # Filter medium clusters
    print("🔍 Filtering medium clusters...")
    filtered_medium_clusters = {}
    original_medium_memos = 0
    filtered_medium_memos = 0
    
    for cluster_id, cluster_data in medium_data['clusters'].items():
        voice_memos = cluster_data['voice_memos']
        original_medium_memos += len(voice_memos)
        
        # Filter voice memos by quality threshold
        filtered_memos = [
            memo for memo in voice_memos 
            if memo.get('quality_coefficient') is not None 
            and isinstance(memo.get('quality_coefficient'), (int, float))
            and memo.get('quality_coefficient') >= QUALITY_THRESHOLD
        ]
        
        # Only include clusters that have voice memos after filtering
        if filtered_memos:
            filtered_medium_memos += len(filtered_memos)
            avg_quality = calculate_average_quality_coefficient(filtered_memos)
            
            filtered_medium_clusters[cluster_id] = {
                "voice_memo_count": len(filtered_memos),
                "average_quality_coefficient": avg_quality,
                "voice_memos": filtered_memos
            }
    
    # Build filtered medium export data
    filtered_medium_data = {
        "export_metadata": {
            "export_type": "condensed_filtered",
            "level": "medium", 
            "exported_at": datetime.now().isoformat(),
            "quality_threshold": QUALITY_THRESHOLD,
            "num_original_clusters": medium_data['export_metadata']['total_clusters'],
            "num_filtered_clusters": len(filtered_medium_clusters),
            "num_clusters_removed": medium_data['export_metadata']['total_clusters'] - len(filtered_medium_clusters),
            "num_original_voice_memos": original_medium_memos,
            "num_filtered_voice_memos": filtered_medium_memos,
            "voice_memos_removed": original_medium_memos - filtered_medium_memos
        },
        "clusters": filtered_medium_clusters
    }
    
    # Save filtered medium clusters
    medium_output_path = output_dir / "filtered_medium_clusters_condensed.json"
    with open(medium_output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_medium_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Filtered medium clusters saved: {medium_output_path}")
    print(f"   📊 {len(filtered_medium_clusters)} clusters ({medium_data['export_metadata']['total_clusters'] - len(filtered_medium_clusters)} removed)")
    print(f"   📊 {filtered_medium_memos} voice memos ({original_medium_memos - filtered_medium_memos} removed)")
    
    # Load and filter coarse clusters
    coarse_input_path = Path("data/04 enriched_clusters/coarse_clusters_condensed.json")
    if not coarse_input_path.exists():
        logger.error(f"Coarse clusters file not found: {coarse_input_path}")
        return 1
        
    print("📂 Loading coarse clusters...")
    with open(coarse_input_path, 'r', encoding='utf-8') as f:
        coarse_data = json.load(f)
    
    # Filter coarse clusters
    print("🔍 Filtering coarse clusters...")
    filtered_coarse_clusters = {}
    original_coarse_memos = 0
    filtered_coarse_memos = 0
    
    for cluster_id, cluster_data in coarse_data['clusters'].items():
        voice_memos = cluster_data['voice_memos']
        original_coarse_memos += len(voice_memos)
        
        # Filter voice memos by quality threshold
        filtered_memos = [
            memo for memo in voice_memos 
            if memo.get('quality_coefficient') is not None 
            and isinstance(memo.get('quality_coefficient'), (int, float))
            and memo.get('quality_coefficient') >= QUALITY_THRESHOLD
        ]
        
        # Only include clusters that have voice memos after filtering
        if filtered_memos:
            filtered_coarse_memos += len(filtered_memos)
            avg_quality = calculate_average_quality_coefficient(filtered_memos)
            
            filtered_coarse_clusters[cluster_id] = {
                "voice_memo_count": len(filtered_memos),
                "average_quality_coefficient": avg_quality,
                "voice_memos": filtered_memos
            }
    
    # Build filtered coarse export data
    filtered_coarse_data = {
        "export_metadata": {
            "export_type": "condensed_filtered",
            "level": "coarse",
            "exported_at": datetime.now().isoformat(),
            "quality_threshold": QUALITY_THRESHOLD,
            "num_original_clusters": coarse_data['export_metadata']['total_clusters'],
            "num_filtered_clusters": len(filtered_coarse_clusters),
            "num_clusters_removed": coarse_data['export_metadata']['total_clusters'] - len(filtered_coarse_clusters),
            "num_original_voice_memos": original_coarse_memos,
            "num_filtered_voice_memos": filtered_coarse_memos,
            "voice_memos_removed": original_coarse_memos - filtered_coarse_memos
        },
        "clusters": filtered_coarse_clusters
    }
    
    # Save filtered coarse clusters
    coarse_output_path = output_dir / "filtered_coarse_clusters_condensed.json"
    with open(coarse_output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_coarse_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Filtered coarse clusters saved: {coarse_output_path}")
    print(f"   📊 {len(filtered_coarse_clusters)} clusters ({coarse_data['export_metadata']['total_clusters'] - len(filtered_coarse_clusters)} removed)")
    print(f"   📊 {filtered_coarse_memos} voice memos ({original_coarse_memos - filtered_coarse_memos} removed)")
    
    print(f"\n🎉 Filtering complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Total Summary:")
    print(f"   • Quality threshold: {QUALITY_THRESHOLD}")
    print(f"   • Medium: {len(filtered_medium_clusters)} clusters, {filtered_medium_memos} voice memos")
    print(f"   • Coarse: {len(filtered_coarse_clusters)} clusters, {filtered_coarse_memos} voice memos")
    
    return 0

def main():
    """Main export process - exports both medium and coarse clusters condensed"""
    print("🗂️  SEMANTIC CLUSTER ENRICHMENT")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data/04 enriched_clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading clustering results...")
    clustering_results = load_clustering_results()
    if not clustering_results:
        return 1
    
    # Export medium clusters with condensed data
    print("📤 Exporting medium clusters (condensed)...")
    medium_cluster_data = export_medium_clusters_condensed(clustering_results)
    if not medium_cluster_data:
        print("❌ Failed to export medium clusters")
        return 1
        
    # Save medium clusters file
    medium_filename = "medium_clusters_condensed.json"
    medium_output_path = output_dir / medium_filename
    
    with open(medium_output_path, 'w', encoding='utf-8') as f:
        json.dump(medium_cluster_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Medium clusters saved: {medium_output_path}")
    print(f"   📊 {medium_cluster_data['export_metadata']['total_clusters']} medium clusters, {medium_cluster_data['export_metadata']['total_voice_memos']} voice memos")
    
    # Export coarse clusters with condensed data
    print("📤 Exporting coarse clusters (condensed)...")
    coarse_cluster_data = export_coarse_clusters_condensed(clustering_results)
    if not coarse_cluster_data:
        print("❌ Failed to export coarse clusters")
        return 1
        
    # Save coarse clusters file
    coarse_filename = "coarse_clusters_condensed.json"
    coarse_output_path = output_dir / coarse_filename
    
    with open(coarse_output_path, 'w', encoding='utf-8') as f:
        json.dump(coarse_cluster_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Coarse clusters saved: {coarse_output_path}")
    print(f"   📊 {coarse_cluster_data['export_metadata']['total_clusters']} coarse clusters, {coarse_cluster_data['export_metadata']['total_voice_memos']} voice memos")
    
    print(f"\n🎉 Enrichment complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Total Summary:")
    print(f"   • {medium_cluster_data['export_metadata']['total_clusters']} medium clusters")
    print(f"   • {coarse_cluster_data['export_metadata']['total_clusters']} coarse clusters") 
    print(f"   • {medium_cluster_data['export_metadata']['total_voice_memos']} total voice memos")
    
    return 0

def create_comprehensive_enriched_clusters() -> int:
    """Create comprehensive enriched clusters combining coarse and medium with descriptions"""
    print("📚 CREATING COMPREHENSIVE ENRICHED CLUSTERS (BOOKS)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data/07 books")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load filtered cluster data
    print("📂 Loading filtered cluster data...")
    try:
        with open("data/05 filtered_enriched_clusters/filtered_coarse_clusters_condensed.json", 'r', encoding='utf-8') as f:
            coarse_data = json.load(f)
        
        with open("data/05 filtered_enriched_clusters/filtered_medium_clusters_condensed.json", 'r', encoding='utf-8') as f:
            medium_data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Required filtered cluster files not found: {e}")
        return 1
    
    # Load cluster descriptions
    print("📂 Loading cluster descriptions...")
    try:
        with open("data/06 cluster_descriptions/coarse.json", 'r', encoding='utf-8') as f:
            coarse_descriptions = json.load(f)
        
        with open("data/06 cluster_descriptions/medium.json", 'r', encoding='utf-8') as f:
            medium_descriptions = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Cluster description files not found: {e}")
        return 1
    
    # Load original cluster assignments to map medium->coarse relationships
    print("📂 Loading cluster assignments for mapping...")
    try:
        clustering_results = load_clustering_results()
        if not clustering_results:
            return 1
    except Exception as e:
        logger.error(f"Failed to load cluster assignments: {e}")
        return 1
    
    # Create mapping from medium cluster ID to coarse cluster ID using original cluster assignments
    print("🗺️  Building medium-to-coarse cluster mapping...")
    medium_to_coarse_map = {}
    
    # Get cluster data from the levels structure
    levels = clustering_results.get('levels', {})
    coarse_clusters = levels.get('coarse', {}).get('clusters', {})
    medium_clusters = levels.get('medium', {}).get('clusters', {})
    
    # Build mapping by finding which coarse cluster each file belongs to, then which medium cluster
    file_to_coarse = {}
    file_to_medium = {}
    
    # Map files to coarse clusters
    for coarse_id, coarse_files in coarse_clusters.items():
        for file_path in coarse_files:
            file_to_coarse[file_path] = coarse_id
    
    # Map files to medium clusters
    for medium_id, medium_files in medium_clusters.items():
        for file_path in medium_files:
            file_to_medium[file_path] = medium_id
    
    # Build medium-to-coarse mapping
    for file_path in file_to_medium:
        medium_id = file_to_medium[file_path]
        coarse_id = file_to_coarse.get(file_path)
        if coarse_id:
            medium_to_coarse_map[medium_id] = coarse_id
    
    print(f"   📊 Built mapping for {len(medium_to_coarse_map)} medium clusters")
    print(f"   📊 Sample mappings: {dict(list(medium_to_coarse_map.items())[:5])}")
    
    # Build comprehensive structure
    print("🏗️  Building comprehensive cluster structure...")
    comprehensive_clusters = {}
    total_voice_memos = 0
    
    # Process each coarse cluster
    for coarse_id, coarse_cluster_data in coarse_data['clusters'].items():
        coarse_voice_memos = coarse_cluster_data['voice_memos']
        coarse_voice_memo_count = coarse_cluster_data['voice_memo_count']
        coarse_avg_quality = coarse_cluster_data['average_quality_coefficient']
        
        # Get coarse cluster description
        coarse_desc = coarse_descriptions.get('cluster_descriptions', {}).get(coarse_id, {})
        coarse_name = coarse_desc.get('name', f'Coarse Cluster {coarse_id}')
        coarse_description = coarse_desc.get('description', 'No description available')
        
        # Find all medium clusters that belong to this coarse cluster
        medium_clusters_in_coarse = {}
        medium_clusters_checked = 0
        medium_clusters_matched = 0
        
        for medium_id, medium_cluster_data in medium_data['clusters'].items():
            medium_clusters_checked += 1
            if medium_to_coarse_map.get(medium_id) == coarse_id:
                medium_clusters_matched += 1
                # Get medium cluster description
                medium_desc = None
                for cluster_info in medium_descriptions.get('clusters', []):
                    if cluster_info.get('cluster_id') == medium_id:
                        medium_desc = cluster_info
                        break
                
                if medium_desc:
                    medium_title = medium_desc.get('title', f'Medium Cluster {medium_id}')
                    medium_raw_essence = medium_desc.get('raw_essence', 'No description available')
                else:
                    medium_title = f'Medium Cluster {medium_id}'
                    medium_raw_essence = 'No description available'
                
                medium_clusters_in_coarse[medium_id] = {
                    "title": medium_title,
                    "raw_essence": medium_raw_essence,
                    "voice_memo_count": medium_cluster_data['voice_memo_count'],
                    "average_quality_coefficient": medium_cluster_data['average_quality_coefficient'],
                    "voice_memos": medium_cluster_data['voice_memos']
                }
        
        comprehensive_clusters[coarse_id] = {
            "name": coarse_name,
            "description": coarse_description,
            "voice_memo_count": coarse_voice_memo_count,
            "average_quality_coefficient": coarse_avg_quality,
            "medium_clusters": medium_clusters_in_coarse
        }
        
        print(f"   📖 Coarse {coarse_id} ({coarse_name}): {medium_clusters_checked} medium checked, {medium_clusters_matched} matched")
        total_voice_memos += coarse_voice_memo_count
    
    # Build export structure
    comprehensive_data = {
        "export_metadata": {
            "export_type": "comprehensive_enriched",
            "exported_at": datetime.now().isoformat(),
            "description": "Comprehensive hierarchical structure: Coarse clusters (books) → Medium clusters (chapters) → Voice memos (pages)",
            "quality_threshold": 0.6,
            "total_coarse_clusters": len(comprehensive_clusters),
            "total_medium_clusters": sum(len(cc['medium_clusters']) for cc in comprehensive_clusters.values()),
            "total_voice_memos": total_voice_memos
        },
        "coarse_clusters": comprehensive_clusters
    }
    
    # Save comprehensive enriched clusters
    output_path = output_dir / "comprehensive_enriched_clusters.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Comprehensive enrichment complete!")
    print(f"📁 File saved: {output_path}")
    print(f"📊 Summary:")
    print(f"   • {len(comprehensive_clusters)} coarse clusters (books)")
    print(f"   • {comprehensive_data['export_metadata']['total_medium_clusters']} medium clusters (chapters)")
    print(f"   • {total_voice_memos} voice memos (pages)")
    print(f"   • Quality threshold: 0.6")
    
    return 0

def get_audio_metadata(audio_path: Path) -> Dict[str, Any]:
    """Extract metadata from audio file"""
    metadata = {
        "file_size_bytes": 0,
        "audio_length_seconds": 0.0,
        "creation_date": None
    }
    
    try:
        # Get file size
        if audio_path.exists():
            stat_result = audio_path.stat()
            metadata["file_size_bytes"] = stat_result.st_size
            
            # Get comprehensive audio metadata using ffprobe JSON output
            try:
                import subprocess
                import json
                
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_format', '-show_streams', str(audio_path)
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0 and result.stdout.strip():
                    ffprobe_data = json.loads(result.stdout)
                    
                    # Get duration
                    if 'format' in ffprobe_data and 'duration' in ffprobe_data['format']:
                        metadata["audio_length_seconds"] = float(ffprobe_data['format']['duration'])
                    
                    # Get creation date from metadata (most accurate)
                    creation_time = None
                    
                    # Try format tags first
                    if 'format' in ffprobe_data and 'tags' in ffprobe_data['format']:
                        format_tags = ffprobe_data['format']['tags']
                        creation_time = format_tags.get('creation_time')
                    
                    # Try stream tags as fallback
                    if not creation_time and 'streams' in ffprobe_data:
                        for stream in ffprobe_data['streams']:
                            if 'tags' in stream:
                                stream_creation = stream['tags'].get('creation_time')
                                if stream_creation:
                                    creation_time = stream_creation
                                    break
                    
                    # Parse creation time if found
                    if creation_time:
                        try:
                            # Handle ISO format with Z suffix
                            if creation_time.endswith('Z'):
                                creation_time = creation_time[:-1] + '+00:00'
                            
                            from datetime import datetime
                            parsed_time = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                            metadata["creation_date"] = parsed_time.isoformat()
                        except ValueError:
                            # Fallback to file system time if parsing fails
                            metadata["creation_date"] = datetime.fromtimestamp(stat_result.st_ctime).isoformat()
                    else:
                        # Fallback to file system time if no creation_time in metadata
                        metadata["creation_date"] = datetime.fromtimestamp(stat_result.st_ctime).isoformat()
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"ffprobe failed for {audio_path}, using fallback: {e}")
                # Fallback: estimate based on file size and use filesystem time
                estimated_minutes = metadata["file_size_bytes"] / (1024 * 1024)  
                metadata["audio_length_seconds"] = estimated_minutes * 60
                metadata["creation_date"] = datetime.fromtimestamp(stat_result.st_ctime).isoformat()
                
    except Exception as e:
        logger.warning(f"Error extracting audio metadata from {audio_path}: {e}")
    
    return metadata

def calculate_text_metadata(transcript: str) -> Dict[str, Any]:
    """Calculate text-based metadata"""
    if not transcript:
        return {
            "word_count": 0,
            "estimated_reading_time_seconds": 0
        }
    
    # Simple word count (split by whitespace)
    word_count = len(transcript.split())
    
    # Estimated reading time: 200 words per minute average
    reading_time_seconds = (word_count / 200) * 60
    
    return {
        "word_count": word_count,
        "estimated_reading_time_seconds": round(reading_time_seconds, 1)
    }

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    try:
        if file_path.exists():
            return file_path.stat().st_size
        return 0
    except Exception:
        return 0

def format_duration_pretty(seconds: float) -> str:
    """Format duration in seconds to pretty spelled-out format"""
    if seconds <= 0:
        return "0 seconds"
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    
    parts = []
    
    if hours > 0:
        hour_text = "hour" if hours == 1 else "hours"
        parts.append(f"{hours} {hour_text}")
    
    if minutes > 0:
        minute_text = "minute" if minutes == 1 else "minutes"
        parts.append(f"{minutes} {minute_text}")
    
    if remaining_seconds > 0:
        second_text = "second" if remaining_seconds == 1 else "seconds"
        parts.append(f"{remaining_seconds} {second_text}")
    
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]}, {parts[1]}"
    else:
        return f"{parts[0]}, {parts[1]}, {parts[2]}"

def format_date_pretty(iso_date_string: str) -> str:
    """Format ISO date string to pretty format like 'August 7, 2025'"""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_date_string.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y")
    except (ValueError, AttributeError):
        return iso_date_string

def calculate_time_span_pretty(start_date: str, end_date: str) -> str:
    """Calculate pretty time span between two dates"""
    try:
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        delta = end_dt - start_dt
        days = delta.days
        
        if days == 0:
            return "Same day"
        elif days < 7:
            return f"{days} day{'s' if days != 1 else ''}"
        elif days < 30:
            weeks = days // 7
            remaining_days = days % 7
            result = f"{weeks} week{'s' if weeks != 1 else ''}"
            if remaining_days > 0:
                result += f", {remaining_days} day{'s' if remaining_days != 1 else ''}"
            return result
        elif days < 365:
            months = days // 30
            remaining_days = days % 30
            result = f"{months} month{'s' if months != 1 else ''}"
            if remaining_days > 0:
                result += f", {remaining_days} day{'s' if remaining_days != 1 else ''}"
            return result
        else:
            years = days // 365
            remaining_days = days % 365
            months = remaining_days // 30
            remaining_days = remaining_days % 30
            
            result = f"{years} year{'s' if years != 1 else ''}"
            if months > 0:
                result += f", {months} month{'s' if months != 1 else ''}"
            if remaining_days > 0:
                result += f", {remaining_days} day{'s' if remaining_days != 1 else ''}"
            return result
            
    except (ValueError, AttributeError):
        return "Unknown span"

# Legacy curved grading functions removed - replaced by assign_curved_af_grades() with full A-F range

def assign_curved_af_grades(quality_scores: List[float]) -> Dict[float, str]:
    """Assign curved A-F grades with +/- modifiers based on percentile ranking"""
    if not quality_scores:
        return {}
    
    # Sort scores in descending order with original quality values
    sorted_scores = sorted(set(quality_scores), reverse=True)
    total = len(sorted_scores)
    
    # Full A-F curved distribution with +/- modifiers
    grade_percentiles = [
        ('A+', 0.05),   # Top 5%
        ('A', 0.08),    # Next 8% 
        ('A-', 0.12),   # Next 12%
        ('B+', 0.15),   # Next 15%
        ('B', 0.20),    # Next 20%
        ('B-', 0.15),   # Next 15%
        ('C+', 0.10),   # Next 10%
        ('C', 0.08),    # Next 8%
        ('C-', 0.05),   # Next 5%
        ('D+', 0.01),   # Next 1%
        ('D', 0.005),   # Next 0.5%
        ('D-', 0.005),  # Next 0.5%
        ('F', 0.0)     # Bottom (calculated as remainder)
    ]
    
    # Calculate cumulative cutoffs
    cutoffs = {}
    cumulative = 0
    for grade, percentage in grade_percentiles[:-1]:  # Skip F, it gets remainder
        cumulative += percentage
        cutoffs[grade] = int(total * cumulative)
    
    # Assign grades based on ranking
    grade_mapping = {}
    current_pos = 0
    for grade, _ in grade_percentiles[:-1]:  # Skip F
        end_pos = cutoffs[grade]
        for i in range(current_pos, min(end_pos, len(sorted_scores))):
            grade_mapping[sorted_scores[i]] = grade
        current_pos = end_pos
    
    # Remaining get F (should be very few or none with this distribution)
    for i in range(current_pos, len(sorted_scores)):
        grade_mapping[sorted_scores[i]] = 'F'
    
    return grade_mapping

def get_af_grade(quality_coefficient: float) -> str:
    """Convert quality coefficient to A-F letter grade with +/- modifiers (absolute thresholds)"""
    if quality_coefficient >= 0.825:
        return "A+"
    elif quality_coefficient >= 0.800:
        return "A"
    elif quality_coefficient >= 0.775:
        return "A-"
    elif quality_coefficient >= 0.750:
        return "B+"
    elif quality_coefficient >= 0.725:
        return "B"
    elif quality_coefficient >= 0.700:
        return "B-"
    elif quality_coefficient >= 0.675:
        return "C+"
    elif quality_coefficient >= 0.650:
        return "C"
    elif quality_coefficient >= 0.625:
        return "C-"
    elif quality_coefficient >= 0.600:
        return "D+"
    elif quality_coefficient >= 0.550:
        return "D"
    elif quality_coefficient >= 0.500:
        return "D-"
    else:
        return "F"

def calculate_composite_book_score(voice_memos: List[Dict], total_vms: int) -> Dict[str, Any]:
    """Calculate composite book scoring with 6-stage taxonomy"""
    import math
    
    if not voice_memos:
        return {
            'volume_factor': 0.0,
            'excellence_count': 0,
            'excellence_pct': 0.0,
            'high_quality_count': 0,
            'high_quality_pct': 0.0,
            'avg_quality': 0.0,
            'avg_quality_factor': 0.0,
            'composite_score': 0.0,
            'pillar_stage': 'FAR_FROM_READY',
            'stage_description': 'No content available',
            'recommended_action': 'Begin exploration and content creation'
        }
    
    # Extract quality coefficients
    qualities = [vm.get('quality_coefficient', 0) for vm in voice_memos if 'quality_coefficient' in vm]
    if not qualities:
        qualities = [0.5]  # Fallback
    
    # Volume factor: Logarithmic scale, plateaus at ~50 VMs
    volume_factor = min(1.0, math.log(total_vms + 1) / math.log(51))
    
    # Excellence count and percentage (quality >= 0.80)
    excellence_count = sum(1 for q in qualities if q >= 0.80)
    excellence_pct = (excellence_count / total_vms) * 100 if total_vms > 0 else 0
    
    # High-quality count and percentage (quality >= 0.75)
    high_quality_count = sum(1 for q in qualities if q >= 0.75)
    high_quality_pct = (high_quality_count / total_vms) * 100 if total_vms > 0 else 0
    
    # Average quality
    avg_quality = sum(qualities) / len(qualities)
    
    # Average quality factor: Normalize 0.60-0.75 range to 0-1
    avg_quality_factor = max(0, (avg_quality - 0.60) / 0.15)
    
    # Composite score: Weighted combination
    composite_score = (
        (excellence_pct / 100) * 0.4 +      # 40% excellence concentration
        (high_quality_pct / 100) * 0.3 +    # 30% high-quality breadth
        volume_factor * 0.2 +               # 20% volume readiness
        avg_quality_factor * 0.1            # 10% average quality
    )
    
    # Determine pillar stage and recommendations
    stage_info = get_pillar_stage_info(total_vms, excellence_pct, high_quality_pct)
    
    return {
        'volume_factor': volume_factor,
        'excellence_count': excellence_count,
        'excellence_pct': excellence_pct,
        'high_quality_count': high_quality_count,
        'high_quality_pct': high_quality_pct,
        'avg_quality': avg_quality,
        'avg_quality_factor': avg_quality_factor,
        'composite_score': composite_score,
        'pillar_stage': stage_info['stage'],
        'stage_description': stage_info['description'],
        'recommended_action': stage_info['action']
    }

def get_pillar_stage_info(total_vms: int, excellence_pct: float, high_quality_pct: float) -> Dict[str, str]:
    """Determine pillar stage and provide recommendations based on 6-stage taxonomy"""
    
    if total_vms >= 50 and excellence_pct >= 12:
        return {
            'stage': 'READY_TO_PUBLISH',
            'description': f'This pillar has substantial content ({total_vms} voice memos) with strong breakthrough density ({excellence_pct:.1f}% excellent insights). The volume and quality combination indicates publication readiness.',
            'action': 'Focus on structuring and editing content for publication. The foundation is solid with consistent breakthrough insights throughout the extensive content base.'
        }
    
    elif (total_vms >= 40 and excellence_pct >= 8 and high_quality_pct >= 35) or (total_vms >= 20 and excellence_pct >= 8 and high_quality_pct >= 40):
        return {
            'stage': 'HIGH_QUALITY',
            'description': f'This pillar shows exceptional insight density with {excellence_pct:.1f}% breakthrough content and {high_quality_pct:.0f}% high-quality thinking across {total_vms} voice memos. There is clearly "something going on here" with concentrated value.',
            'action': 'Expand the successful patterns that are generating breakthroughs. The quality foundation is strong - focus on generating more content in the areas that are producing exceptional insights.'
        }
    
    elif total_vms >= 40 and excellence_pct >= 6:
        return {
            'stage': 'BUILDING_MOMENTUM',
            'description': f'This pillar has a solid foundation with {total_vms} voice memos and emerging breakthrough patterns ({excellence_pct:.1f}% excellent). The groundwork is established and quality signals are developing.',
            'action': 'Focus on breakthrough development over volume expansion. Identify what conditions or topics generate the excellent insights and deliberately create more content in those areas.'
        }
    
    elif excellence_pct >= 8 or (total_vms >= 20 and high_quality_pct >= 30):
        return {
            'stage': 'FIRST_BREAKTHROUGH',
            'description': f'This pillar shows clear breakthrough signals with {excellence_pct:.1f}% excellent insights despite limited volume ({total_vms} voice memos). The quality indicators suggest strong potential.',
            'action': 'Generate more content specifically in the breakthrough areas. The excellence signals indicate you have found valuable patterns - now scale up content creation around these successful themes.'
        }
    
    elif excellence_pct >= 4 or high_quality_pct >= 20:
        return {
            'stage': 'EARLY_SIGNS',
            'description': f'This pillar shows emerging quality signals with {high_quality_pct:.0f}% high-quality content across {total_vms} voice memos. Quality patterns are beginning to appear but remain inconsistent.',
            'action': 'Continue exploring to identify what generates the quality insights. Focus on recognizing patterns in the higher-quality content to understand what conditions produce better thinking.'
        }
    
    else:
        return {
            'stage': 'FAR_FROM_READY',
            'description': f'This pillar is in early exploration with {total_vms} voice memos showing limited breakthrough density ({excellence_pct:.1f}% excellent). The foundation is being established but quality patterns are not yet clear.',
            'action': 'Focus on quality over quantity in content creation. Experiment with different approaches, topics, and conditions to discover what generates insights worth developing further.'
        }

def calculate_temporal_metadata(creation_dates: list) -> Dict[str, Any]:
    """Calculate temporal metadata from a list of creation dates"""
    if not creation_dates:
        return {
            "first_entry_date": None,
            "first_entry_pretty": None,
            "latest_entry_date": None,
            "latest_entry_pretty": None,
            "time_span_days": 0,
            "time_span_pretty": None
        }
    
    # Sort dates to get first and last
    valid_dates = [date for date in creation_dates if date]
    if not valid_dates:
        return {
            "first_entry_date": None,
            "first_entry_pretty": None,
            "latest_entry_date": None,
            "latest_entry_pretty": None,
            "time_span_days": 0,
            "time_span_pretty": None
        }
    
    sorted_dates = sorted(valid_dates)
    first_date = sorted_dates[0]
    last_date = sorted_dates[-1]
    
    # Calculate span
    try:
        from datetime import datetime
        first_dt = datetime.fromisoformat(first_date.replace('Z', '+00:00'))
        last_dt = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
        span_days = (last_dt - first_dt).days
    except (ValueError, AttributeError):
        span_days = 0
    
    return {
        "first_entry_date": first_date,
        "first_entry_pretty": format_date_pretty(first_date),
        "latest_entry_date": last_date,
        "latest_entry_pretty": format_date_pretty(last_date),
        "time_span_days": span_days,
        "time_span_pretty": calculate_time_span_pretty(first_date, last_date)
    }

def create_notion_library_export() -> int:
    """Create final Notion export with one JSON file per book"""
    print("📚 CREATING NOTION LIBRARY EXPORT")
    print("=" * 60)
    
    # First, collect all quality scores to create curved grade mappings
    print("📊 Analyzing quality distributions for curved grading...")
    voice_memo_scores = []
    cluster_scores = []
    chapter_scores = []
    book_scores = []
    
    # Load required data first for collection pass
    try:
        with open("data/07 books/chapter_cluster_assigment.json", 'r', encoding='utf-8') as f:
            chapter_assignments = json.load(f)
        
        with open("data/07 books/comprehensive_enriched_clusters.json", 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        with open("data/06 cluster_descriptions/medium.json", 'r', encoding='utf-8') as f:
            cluster_descriptions = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        return 1
    
    coarse_clusters_data = comprehensive_data.get('coarse_clusters', {})
    
    # Create cluster description lookup
    cluster_desc_lookup = {}
    for cluster_data in cluster_descriptions.get('clusters', []):
        cluster_id = str(cluster_data.get('cluster_id', ''))
        cluster_desc_lookup[cluster_id] = cluster_data
    
    # Collection pass: gather all quality scores
    for book_info in chapter_assignments:
        book_title = book_info['book_title']
        
        # Find corresponding coarse cluster data
        book_coarse_data = None
        for coarse_id, coarse_data in coarse_clusters_data.items():
            if coarse_data.get('name', '').strip() == book_title.strip():
                book_coarse_data = coarse_data
                break
        
        if not book_coarse_data:
            continue
        
        book_quality_scores_temp = []
        
        for chapter_info in book_info['chapters']:
            chapter_quality_scores_temp = []
            
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                
                # Find cluster in comprehensive data
                cluster_voice_memos = []
                for coarse_id, coarse_data in coarse_clusters_data.items():
                    if cluster_id_str in coarse_data.get('medium_clusters', {}):
                        cluster_data = coarse_data['medium_clusters'][cluster_id_str]
                        cluster_voice_memos = cluster_data.get('voice_memos', [])
                        break
                
                if not cluster_voice_memos:
                    continue
                
                # Collect voice memo scores for this cluster
                cluster_vm_scores = []
                for voice_memo in cluster_voice_memos:
                    filename = voice_memo['filename']
                    fingerprint = load_fingerprint(filename)
                    
                    if fingerprint and 'insight_quality' in fingerprint:
                        quality_coefficient = fingerprint['insight_quality'].get('quality_coefficient', 0.0)
                        voice_memo_scores.append(quality_coefficient)
                        cluster_vm_scores.append(quality_coefficient)
                
                # Calculate cluster average
                if cluster_vm_scores:
                    cluster_avg = sum(cluster_vm_scores) / len(cluster_vm_scores)
                    cluster_scores.append(cluster_avg)
                    chapter_quality_scores_temp.extend(cluster_vm_scores)
            
            # Calculate chapter average
            if chapter_quality_scores_temp:
                chapter_avg = sum(chapter_quality_scores_temp) / len(chapter_quality_scores_temp)
                chapter_scores.append(chapter_avg)
                book_quality_scores_temp.extend(chapter_quality_scores_temp)
        
        # Calculate book average
        if book_quality_scores_temp:
            book_avg = sum(book_quality_scores_temp) / len(book_quality_scores_temp)
            book_scores.append(book_avg)
    
    # Create curved grade mappings for each level
    print(f"   • Voice memos: {len(voice_memo_scores)} scores")
    print(f"   • Clusters: {len(cluster_scores)} scores")
    print(f"   • Chapters: {len(chapter_scores)} scores")  
    print(f"   • Books: {len(book_scores)} scores")
    
    voice_memo_grade_mapping = assign_curved_grades(voice_memo_scores)
    cluster_grade_mapping = assign_curved_grades(cluster_scores)
    chapter_grade_mapping = assign_curved_grades(chapter_scores)
    book_grade_mapping = assign_curved_grades(book_scores)
    
    # Create output directory
    output_dir = Path("data/08 library")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Now do the actual export with curved grades
    print("📚 Exporting library with curved grades...")
    
    # Process each book
    books_processed = 0
    for book_info in chapter_assignments:
        book_title = book_info['book_title']
        print(f"\n📖 Processing book: {book_title}")
        
        # Find matching coarse cluster data
        book_coarse_data = None
        for coarse_id, coarse_data in coarse_clusters_data.items():
            if coarse_data['name'].lower() == book_title.lower():
                book_coarse_data = coarse_data
                break
        
        if not book_coarse_data:
            print(f"   ⚠️  Warning: No coarse cluster data found for book '{book_title}'")
            continue
        
        # Build book structure
        book_data = {
            "book_metadata": {
                "title": book_title,
                "description": book_coarse_data.get('description', ''),
                "total_chapters": len(book_info['chapters']),
                "total_clusters": 0,
                "total_voice_memos": 0,
                "average_quality_coefficient": book_coarse_data.get('average_quality_coefficient', 0),
                "total_file_size_bytes": 0,
                "total_reading_time_minutes": 0,
                "total_audio_length_seconds": 0
            },
            "chapters": []
        }
        
        # Initialize book-level aggregation variables
        book_quality_scores = []
        book_creation_dates = []
        
        # Process chapters
        for chapter_info in book_info['chapters']:
            chapter_data = {
                "chapter_title": chapter_info['chapter_title'],
                "part_title": chapter_info['part_title'],
                "chapter_metadata": {
                    "cluster_count": len(chapter_info['clusters']),
                    "voice_memo_count": 0,
                    "average_quality_coefficient": 0,
                    "total_reading_time_minutes": 0,
                    "total_audio_length_seconds": 0
                },
                "clusters": []
            }
            
            chapter_quality_scores = []
            chapter_reading_time = 0
            chapter_audio_length = 0
            chapter_voice_memo_count = 0
            chapter_creation_dates = []
            
            # Process clusters in chapter
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                
                # Find cluster in comprehensive data
                cluster_found = False
                cluster_voice_memos = []
                
                for coarse_id, coarse_data in coarse_clusters_data.items():
                    if cluster_id_str in coarse_data.get('medium_clusters', {}):
                        cluster_data = coarse_data['medium_clusters'][cluster_id_str]
                        cluster_voice_memos = cluster_data.get('voice_memos', [])
                        cluster_found = True
                        break
                
                if not cluster_found:
                    print(f"   ⚠️  Warning: Cluster {cluster_id} not found in comprehensive data")
                    continue
                
                # Get cluster description
                cluster_desc = cluster_desc_lookup.get(cluster_id_str, {})
                cluster_name = cluster_desc.get('title', f'Cluster {cluster_id}')
                cluster_description = cluster_desc.get('raw_essence', 'No description available')
                
                # Process voice memos with rich metadata
                enriched_voice_memos = []
                cluster_quality_scores = []
                cluster_reading_time = 0
                cluster_audio_length = 0
                
                for voice_memo in cluster_voice_memos:
                    filename = voice_memo['filename']
                    
                    # Load full transcript and fingerprint
                    transcript = load_transcript(filename)
                    fingerprint = load_fingerprint(filename)
                    
                    if not transcript and not fingerprint:
                        continue
                    
                    # Calculate text metadata
                    text_metadata = calculate_text_metadata(transcript or "")
                    
                    # Get file sizes
                    transcript_path = Path("data/00 transcripts") / filename.replace('.json', '.txt')
                    fingerprint_path = Path("data/01 fingerprints") / filename
                    
                    # Get audio metadata  
                    audio_filename = filename.replace('.json', '.m4a')
                    audio_path = Path("audio_files") / audio_filename
                    audio_metadata = get_audio_metadata(audio_path)
                    
                    # Get quality coefficient for ranking
                    quality_coefficient = 0.0
                    if fingerprint and 'insight_quality' in fingerprint:
                        quality_coefficient = fingerprint['insight_quality'].get('quality_coefficient', 0.0)
                    
                    quality_ranking = get_quality_ranking(quality_coefficient, "voice_memo", voice_memo_grade_mapping)
                    
                    # Combine all metadata
                    voice_memo_metadata = {
                        "file_size_bytes": (
                            get_file_size(transcript_path) + 
                            get_file_size(fingerprint_path) + 
                            audio_metadata["file_size_bytes"]
                        ),
                        "word_count": text_metadata["word_count"],
                        "estimated_reading_time_seconds": text_metadata["estimated_reading_time_seconds"],
                        "estimated_reading_time_pretty": format_duration_pretty(text_metadata["estimated_reading_time_seconds"]),
                        "audio_length_seconds": audio_metadata["audio_length_seconds"],
                        "audio_length_pretty": format_duration_pretty(audio_metadata["audio_length_seconds"]),
                        "creation_date": audio_metadata["creation_date"],
                        "creation_date_pretty": format_date_pretty(audio_metadata["creation_date"]) if audio_metadata["creation_date"] else None,
                        "quality_coefficient": quality_coefficient,
                        "quality_grade": quality_ranking
                    }
                    
                    enriched_voice_memo = {
                        "filename": filename,
                        "transcript": transcript or "",
                        "fingerprint": fingerprint or {},
                        "metadata": voice_memo_metadata
                    }
                    
                    enriched_voice_memos.append(enriched_voice_memo)
                    
                    # Aggregate for cluster
                    if fingerprint and fingerprint.get('insight_quality', {}).get('quality_coefficient'):
                        cluster_quality_scores.append(fingerprint['insight_quality']['quality_coefficient'])
                    cluster_reading_time += voice_memo_metadata["estimated_reading_time_seconds"]
                    cluster_audio_length += voice_memo_metadata["audio_length_seconds"]
                
                # Calculate temporal metadata for cluster
                cluster_creation_dates = [vm["metadata"]["creation_date"] for vm in enriched_voice_memos if vm["metadata"]["creation_date"]]
                cluster_temporal_metadata = calculate_temporal_metadata(cluster_creation_dates)
                
                # Calculate cluster quality ranking
                avg_quality = sum(cluster_quality_scores) / len(cluster_quality_scores) if cluster_quality_scores else 0
                cluster_quality_ranking = get_quality_ranking(avg_quality, "cluster", cluster_grade_mapping)
                
                # Build cluster data
                cluster_output = {
                    "cluster_id": cluster_id_str,
                    "cluster_name": cluster_name,
                    "cluster_description": cluster_description,
                    "cluster_metadata": {
                        "voice_memo_count": len(enriched_voice_memos),
                        "average_quality_coefficient": avg_quality,
                        "quality_grade": cluster_quality_ranking,
                        "total_reading_time_minutes": round(cluster_reading_time / 60, 2),
                        "total_reading_time_pretty": format_duration_pretty(cluster_reading_time),
                        "total_audio_length_seconds": round(cluster_audio_length, 1),
                        "total_audio_length_pretty": format_duration_pretty(cluster_audio_length),
                        **cluster_temporal_metadata
                    },
                    "voice_memos": enriched_voice_memos
                }
                
                chapter_data["clusters"].append(cluster_output)
                
                # Aggregate for chapter
                chapter_quality_scores.extend(cluster_quality_scores)
                chapter_reading_time += cluster_reading_time
                chapter_audio_length += cluster_audio_length
                chapter_voice_memo_count += len(enriched_voice_memos)
                chapter_creation_dates.extend(cluster_creation_dates)
            
            # Calculate chapter temporal metadata and quality ranking
            chapter_temporal_metadata = calculate_temporal_metadata(chapter_creation_dates)
            chapter_avg_quality = sum(chapter_quality_scores) / len(chapter_quality_scores) if chapter_quality_scores else 0
            chapter_quality_ranking = get_quality_ranking(chapter_avg_quality, "chapter", chapter_grade_mapping)
            
            # Update chapter metadata
            chapter_data["chapter_metadata"].update({
                "voice_memo_count": chapter_voice_memo_count,
                "average_quality_coefficient": chapter_avg_quality,
                "quality_grade": chapter_quality_ranking,
                "total_reading_time_minutes": round(chapter_reading_time / 60, 2),
                "total_reading_time_pretty": format_duration_pretty(chapter_reading_time),
                "total_audio_length_seconds": round(chapter_audio_length, 1),
                "total_audio_length_pretty": format_duration_pretty(chapter_audio_length),
                **chapter_temporal_metadata
            })
            
            book_data["chapters"].append(chapter_data)
            
            # Aggregate for book
            book_data["book_metadata"]["total_clusters"] += len(chapter_data["clusters"])
            book_data["book_metadata"]["total_voice_memos"] += chapter_voice_memo_count
            book_data["book_metadata"]["total_reading_time_minutes"] += chapter_data["chapter_metadata"]["total_reading_time_minutes"]
            book_data["book_metadata"]["total_audio_length_seconds"] += chapter_data["chapter_metadata"]["total_audio_length_seconds"]
            book_quality_scores.extend(chapter_quality_scores)
            book_creation_dates.extend(chapter_creation_dates)
        
        # Calculate final book-level metadata
        book_temporal_metadata = calculate_temporal_metadata(book_creation_dates)
        book_avg_quality = sum(book_quality_scores) / len(book_quality_scores) if book_quality_scores else 0
        book_quality_ranking = get_quality_ranking(book_avg_quality, "book", book_grade_mapping)
        
        # Update book metadata with enhanced information
        book_data["book_metadata"].update({
            "average_quality_coefficient": book_avg_quality,
            "quality_grade": book_quality_ranking,
            "total_reading_time_pretty": format_duration_pretty(book_data["book_metadata"]["total_reading_time_minutes"] * 60),
            "total_audio_length_pretty": format_duration_pretty(book_data["book_metadata"]["total_audio_length_seconds"]),
            **book_temporal_metadata
        })
        
        # Save book file
        book_filename = f"{book_title}.json"
        book_output_path = output_dir / book_filename
        
        with open(book_output_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Exported: {book_filename}")
        print(f"   📊 {book_data['book_metadata']['total_chapters']} chapters, "
              f"{book_data['book_metadata']['total_clusters']} clusters, "
              f"{book_data['book_metadata']['total_voice_memos']} voice memos")
        
        books_processed += 1
    
    print(f"\n🎉 Notion library export complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Summary: {books_processed} books exported")
    
    return 0

def create_enhanced_full_library_export() -> int:
    """Create full library export with ALL voice memos, curved A-F grading, and composite book scoring"""
    print("📚 CREATING ENHANCED FULL LIBRARY EXPORT")
    print("=" * 60)
    print("🎯 Features: Curved A-F grading + Composite book scoring + 6-stage taxonomy")
    
    # Create output directory
    output_dir = Path("data/09 library full enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load required data
    print("📂 Loading all data sources...")
    try:
        # Load chapter assignments
        with open("data/07 books/chapter_cluster_assigment.json", 'r', encoding='utf-8') as f:
            chapter_assignments = json.load(f)
        
        # Load ALL enriched clusters (unfiltered)
        with open("data/04 enriched_clusters/medium_clusters_condensed.json", 'r', encoding='utf-8') as f:
            all_enriched_data = json.load(f)
        
        # Load comprehensive data for structure
        with open("data/07 books/comprehensive_enriched_clusters.json", 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        # Load cluster descriptions
        with open("data/06 cluster_descriptions/medium.json", 'r', encoding='utf-8') as f:
            cluster_descriptions = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        return 1
    
    # Create lookup dictionaries
    cluster_desc_lookup = {}
    for cluster_data in cluster_descriptions.get('clusters', []):
        cluster_id = str(cluster_data.get('cluster_id', ''))
        cluster_desc_lookup[cluster_id] = cluster_data
    
    # Create lookup for ALL voice memos by filename
    all_voice_memos = {}
    for cluster_id, cluster_data in all_enriched_data.get('clusters', {}).items():
        for voice_memo in cluster_data.get('voice_memos', []):
            filename = voice_memo['filename']
            all_voice_memos[filename] = voice_memo
    
    print(f"📈 Processing {len(all_voice_memos)} total voice memos")
    
    # STEP 1: Collect all quality scores for curved grading
    print("📈 Step 1: Collecting quality scores for curved grading...")
    voice_memo_qualities = []
    cluster_avg_qualities = []
    chapter_avg_qualities = []
    
    # Collect voice memo qualities and calculate cluster/chapter averages
    for book_info in chapter_assignments:
        for chapter_info in book_info['chapters']:
            chapter_qualities = []
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                if cluster_id_str in all_enriched_data.get('clusters', {}):
                    cluster_qualities = []
                    for vm in all_enriched_data['clusters'][cluster_id_str].get('voice_memos', []):
                        quality_coefficient = vm.get('quality_coefficient', 0.0)
                        if quality_coefficient > 0:
                            voice_memo_qualities.append(quality_coefficient)
                            cluster_qualities.append(quality_coefficient)
                    
                    # Calculate cluster average
                    if cluster_qualities:
                        cluster_avg = sum(cluster_qualities) / len(cluster_qualities)
                        cluster_avg_qualities.append(cluster_avg)
                        chapter_qualities.extend(cluster_qualities)
            
            # Calculate chapter average  
            if chapter_qualities:
                chapter_avg = sum(chapter_qualities) / len(chapter_qualities)
                chapter_avg_qualities.append(chapter_avg)
    
    # Generate curved grade mappings for all levels
    print(f"   🎢 Voice memos: {len(voice_memo_qualities)} scores")
    print(f"   🎢 Clusters: {len(cluster_avg_qualities)} scores")
    print(f"   🎢 Chapters: {len(chapter_avg_qualities)} scores")
    
    vm_grade_mapping = assign_curved_af_grades(voice_memo_qualities)
    cluster_grade_mapping = assign_curved_af_grades(cluster_avg_qualities)
    chapter_grade_mapping = assign_curved_af_grades(chapter_avg_qualities)
    
    coarse_clusters_data = comprehensive_data.get('coarse_clusters', {})
    
    # STEP 2: Process each book with enhanced scoring
    print("📈 Step 2: Processing books with enhanced scoring...")
    books_processed = 0
    
    for book_info in chapter_assignments:
        book_title = book_info['book_title']
        print(f"\n📖 Processing book: {book_title}")
        
        # Find matching coarse cluster data
        book_coarse_data = None
        for coarse_id, coarse_data in coarse_clusters_data.items():
            if coarse_data.get('name', '').strip() == book_title.strip():
                book_coarse_data = coarse_data
                break
        
        if not book_coarse_data:
            print(f"   ⚠️  Warning: No coarse cluster data found for book '{book_title}'")
            continue
        
        # Build book structure with enhanced metadata
        book_data = {
            "book_metadata": {
                "title": book_title,
                "description": book_coarse_data.get('description', ''),
                "total_chapters": len(book_info['chapters']),
                "total_clusters": 0,
                "total_voice_memos": 0,
                "total_file_size_bytes": 0,
                "total_reading_time_minutes": 0,
                "total_audio_length_seconds": 0
            },
            "chapters": []
        }
        
        # Initialize book-level aggregation
        book_quality_scores = []
        book_creation_dates = []
        all_book_voice_memos = []  # For composite scoring
        
        # Process chapters
        for chapter_info in book_info['chapters']:
            chapter_data = {
                "chapter_title": chapter_info['chapter_title'],
                "part_title": chapter_info['part_title'],
                "chapter_metadata": {
                    "cluster_count": len(chapter_info['clusters']),
                },
                "clusters": []
            }
            
            chapter_quality_scores = []
            chapter_reading_time = 0
            chapter_audio_length = 0
            chapter_voice_memo_count = 0
            chapter_creation_dates = []
            
            # Process clusters in chapter
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                
                # Get cluster description
                cluster_desc = cluster_desc_lookup.get(cluster_id_str, {})
                cluster_name = cluster_desc.get('title', f'Cluster {cluster_id}')
                cluster_description = cluster_desc.get('raw_essence', 'No description available')
                
                # Find ALL voice memos for this cluster
                cluster_voice_memos = []
                if cluster_id_str in all_enriched_data.get('clusters', {}):
                    for vm in all_enriched_data['clusters'][cluster_id_str].get('voice_memos', []):
                        if vm['filename'] in all_voice_memos:
                            cluster_voice_memos.append(vm['filename'])
                
                if not cluster_voice_memos:
                    continue
                
                # Process voice memos with curved grading
                enriched_voice_memos = []
                cluster_quality_scores = []
                cluster_reading_time = 0
                cluster_audio_length = 0
                
                for filename in cluster_voice_memos:
                    voice_memo = all_voice_memos[filename]
                    
                    # Load full transcript and fingerprint
                    transcript = load_transcript(filename)
                    fingerprint = load_fingerprint(filename)
                    
                    if not transcript and not fingerprint:
                        continue
                    
                    # Calculate text metadata
                    text_metadata = calculate_text_metadata(transcript or "")
                    
                    # Get audio metadata
                    audio_filename = filename.replace('.json', '.m4a')
                    audio_path = Path("audio_files") / audio_filename
                    audio_metadata = get_audio_metadata(audio_path)
                    
                    # Get quality coefficient for curved grading
                    quality_coefficient = voice_memo.get('quality_coefficient', 0.0)
                    if fingerprint and 'insight_quality' in fingerprint:
                        quality_coefficient = fingerprint['insight_quality'].get('quality_coefficient', 0.0)
                    
                    # Apply curved grading
                    curved_grade = vm_grade_mapping.get(quality_coefficient, "B+")
                    
                    # Combine all metadata
                    voice_memo_metadata = {
                        "file_size_bytes": (
                            get_file_size(Path("data/00 transcripts") / filename.replace('.json', '.txt')) + 
                            get_file_size(Path("data/01 fingerprints") / filename) + 
                            audio_metadata["file_size_bytes"]
                        ),
                        "word_count": text_metadata["word_count"],
                        "estimated_reading_time_seconds": text_metadata["estimated_reading_time_seconds"],
                        "estimated_reading_time_pretty": format_duration_pretty(text_metadata["estimated_reading_time_seconds"]),
                        "audio_length_seconds": audio_metadata["audio_length_seconds"],
                        "audio_length_pretty": format_duration_pretty(audio_metadata["audio_length_seconds"]),
                        "creation_date": audio_metadata["creation_date"],
                        "creation_date_pretty": format_date_pretty(audio_metadata["creation_date"]) if audio_metadata["creation_date"] else None,
                        "quality_coefficient": quality_coefficient,
                        "quality_grade_curved": curved_grade
                    }
                    
                    enriched_voice_memo = {
                        "filename": filename,
                        "transcript": transcript or "",
                        "fingerprint": fingerprint or {},
                        "metadata": voice_memo_metadata
                    }
                    
                    enriched_voice_memos.append(enriched_voice_memo)
                    all_book_voice_memos.append(voice_memo_metadata)  # For book composite scoring
                    
                    # Aggregate for cluster
                    if quality_coefficient:
                        cluster_quality_scores.append(quality_coefficient)
                    cluster_reading_time += voice_memo_metadata["estimated_reading_time_seconds"]
                    cluster_audio_length += voice_memo_metadata["audio_length_seconds"]
                    if voice_memo_metadata["creation_date"]:
                        chapter_creation_dates.append(voice_memo_metadata["creation_date"])
                
                # Calculate cluster metadata with curved grading
                cluster_temporal_metadata = calculate_temporal_metadata([vm["metadata"]["creation_date"] for vm in enriched_voice_memos if vm["metadata"]["creation_date"]])
                cluster_avg_quality = sum(cluster_quality_scores) / len(cluster_quality_scores) if cluster_quality_scores else 0
                cluster_curved_grade = cluster_grade_mapping.get(cluster_avg_quality, "B+")
                
                # Build cluster data
                cluster_output = {
                    "cluster_id": cluster_id_str,
                    "cluster_name": cluster_name,
                    "cluster_description": cluster_description,
                    "cluster_metadata": {
                        "voice_memo_count": len(enriched_voice_memos),
                        "average_quality_coefficient": cluster_avg_quality,
                        "quality_grade_curved": cluster_curved_grade,
                        "total_reading_time_minutes": round(cluster_reading_time / 60, 2),
                        "total_reading_time_pretty": format_duration_pretty(cluster_reading_time),
                        "total_audio_length_seconds": round(cluster_audio_length, 1),
                        "total_audio_length_pretty": format_duration_pretty(cluster_audio_length),
                        **cluster_temporal_metadata
                    },
                    "voice_memos": enriched_voice_memos
                }
                
                chapter_data["clusters"].append(cluster_output)
                
                # Aggregate for chapter
                chapter_quality_scores.extend(cluster_quality_scores)
                chapter_reading_time += cluster_reading_time
                chapter_audio_length += cluster_audio_length
                chapter_voice_memo_count += len(enriched_voice_memos)
            
            # Calculate chapter metadata with curved grading
            chapter_temporal_metadata = calculate_temporal_metadata(chapter_creation_dates)
            chapter_avg_quality = sum(chapter_quality_scores) / len(chapter_quality_scores) if chapter_quality_scores else 0
            chapter_curved_grade = chapter_grade_mapping.get(chapter_avg_quality, "B+")
            
            # Update chapter metadata
            chapter_data["chapter_metadata"].update({
                "voice_memo_count": chapter_voice_memo_count,
                "average_quality_coefficient": chapter_avg_quality,
                "quality_grade_curved": chapter_curved_grade,
                "total_reading_time_minutes": round(chapter_reading_time / 60, 2),
                "total_reading_time_pretty": format_duration_pretty(chapter_reading_time),
                "total_audio_length_seconds": round(chapter_audio_length, 1),
                "total_audio_length_pretty": format_duration_pretty(chapter_audio_length),
                **chapter_temporal_metadata
            })
            
            book_data["chapters"].append(chapter_data)
            
            # Aggregate for book
            book_data["book_metadata"]["total_clusters"] += len(chapter_data["clusters"])
            book_data["book_metadata"]["total_voice_memos"] += chapter_voice_memo_count
            book_data["book_metadata"]["total_reading_time_minutes"] += chapter_data["chapter_metadata"]["total_reading_time_minutes"]
            book_data["book_metadata"]["total_audio_length_seconds"] += chapter_data["chapter_metadata"]["total_audio_length_seconds"]
            book_quality_scores.extend(chapter_quality_scores)
            book_creation_dates.extend(chapter_creation_dates)
        
        # Calculate comprehensive book metadata with composite scoring
        book_temporal_metadata = calculate_temporal_metadata(book_creation_dates)
        book_avg_quality = sum(book_quality_scores) / len(book_quality_scores) if book_quality_scores else 0
        
        # Calculate composite book score with all sub-scores
        composite_scoring = calculate_composite_book_score(
            all_book_voice_memos,
            book_data["book_metadata"]["total_voice_memos"]
        )
        
        # Update book metadata with comprehensive scoring
        book_data["book_metadata"].update({
            "average_quality_coefficient": book_avg_quality,
            "total_reading_time_pretty": format_duration_pretty(book_data["book_metadata"]["total_reading_time_minutes"] * 60),
            "total_audio_length_pretty": format_duration_pretty(book_data["book_metadata"]["total_audio_length_seconds"]),
            
            # Composite scoring components (detailed breakdown)
            "volume_factor": round(composite_scoring["volume_factor"], 3),
            "excellence_count": composite_scoring["excellence_count"],
            "excellence_percentage": round(composite_scoring["excellence_pct"], 1),
            "high_quality_count": composite_scoring["high_quality_count"],
            "high_quality_percentage": round(composite_scoring["high_quality_pct"], 1),
            "average_quality_factor": round(composite_scoring["avg_quality_factor"], 3),
            "composite_score": round(composite_scoring["composite_score"], 3),
            
            # 6-stage pillar taxonomy and detailed recommendations
            "pillar_stage": composite_scoring["pillar_stage"],
            "stage_description": composite_scoring["stage_description"],
            "recommended_action": composite_scoring["recommended_action"],
            
            **book_temporal_metadata
        })
        
        # Save book file
        book_filename = f"{book_title}.json"
        book_output_path = output_dir / book_filename
        
        with open(book_output_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Exported: {book_filename}")
        print(f"   📈 {book_data['book_metadata']['total_chapters']} chapters, "
              f"{book_data['book_metadata']['total_clusters']} clusters, "
              f"{book_data['book_metadata']['total_voice_memos']} voice memos")
        print(f"   🏆 Stage: {composite_scoring['pillar_stage']} (Score: {composite_scoring['composite_score']:.3f})")
        print(f"   📊 Excellence: {composite_scoring['excellence_count']} insights ({composite_scoring['excellence_pct']:.1f}%)")
        
        books_processed += 1
    
    print(f"\n🎉 Enhanced full library export complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📈 Summary: {books_processed} books with enhanced scoring")
    print(f"🎢 Curved A-F grading applied at all levels")
    print(f"🎯 Composite book scoring with 6-stage taxonomy")
    
    return 0

def create_full_library_export() -> int:
    """Legacy function - use create_enhanced_full_library_export() for new features"""
    return create_enhanced_full_library_export()
    
    # Create output directory
    output_dir = Path("data/09 library full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load required data
    print("📂 Loading all data sources...")
    try:
        # Load chapter assignments
        with open("data/07 books/chapter_cluster_assigment.json", 'r', encoding='utf-8') as f:
            chapter_assignments = json.load(f)
        
        # Load ALL enriched clusters (unfiltered)
        with open("data/04 enriched_clusters/medium_clusters_condensed.json", 'r', encoding='utf-8') as f:
            all_enriched_data = json.load(f)
        
        # Load comprehensive data for structure
        with open("data/07 books/comprehensive_enriched_clusters.json", 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        # Load cluster descriptions
        with open("data/06 cluster_descriptions/medium.json", 'r', encoding='utf-8') as f:
            cluster_descriptions = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        return 1
    
    # Create lookup dictionaries
    cluster_desc_lookup = {}
    for cluster_data in cluster_descriptions.get('clusters', []):
        cluster_id = str(cluster_data.get('cluster_id', ''))
        cluster_desc_lookup[cluster_id] = cluster_data
    
    # Create lookup for ALL voice memos by filename
    all_voice_memos = {}
    for cluster_id, cluster_data in all_enriched_data.get('clusters', {}).items():
        for voice_memo in cluster_data.get('voice_memos', []):
            filename = voice_memo['filename']
            all_voice_memos[filename] = voice_memo
    
    print(f"📊 Processing {len(all_voice_memos)} total voice memos")
    
    coarse_clusters_data = comprehensive_data.get('coarse_clusters', {})
    
    # Collect all quality scores for curved grading
    print("📈 Collecting quality scores for curved grading...")
    all_quality_scores = []
    
    for book_info in chapter_assignments:
        for chapter_info in book_info['chapters']:
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                if cluster_id_str in all_enriched_data.get('clusters', {}):
                    for vm in all_enriched_data['clusters'][cluster_id_str].get('voice_memos', []):
                        quality_coefficient = vm.get('quality_coefficient', 0.0)
                        if quality_coefficient > 0:
                            all_quality_scores.append(quality_coefficient)
    
    # Generate curved grade mappings for all levels
    print(f"   🎢 Generating curved grades for {len(all_quality_scores)} voice memos")
    voice_memo_grade_mapping = assign_curved_af_grades(all_quality_scores)
    
    # Process each book
    books_processed = 0
    for book_info in chapter_assignments:
        book_title = book_info['book_title']
        print(f"\n📖 Processing book: {book_title}")
        
        # Find matching coarse cluster data
        book_coarse_data = None
        for coarse_id, coarse_data in coarse_clusters_data.items():
            if coarse_data.get('name', '').strip() == book_title.strip():
                book_coarse_data = coarse_data
                break
        
        if not book_coarse_data:
            print(f"   ⚠️  Warning: No coarse cluster data found for book '{book_title}'")
            continue
        
        # Build book structure
        book_data = {
            "book_metadata": {
                "title": book_title,
                "description": book_coarse_data.get('description', ''),
                "total_chapters": len(book_info['chapters']),
                "total_clusters": 0,
                "total_voice_memos": 0,
                "total_file_size_bytes": 0,
                "total_reading_time_minutes": 0,
                "total_audio_length_seconds": 0
            },
            "chapters": []
        }
        
        # Initialize book-level aggregation
        book_quality_scores = []
        book_creation_dates = []
        
        # Process chapters
        for chapter_info in book_info['chapters']:
            chapter_data = {
                "chapter_title": chapter_info['chapter_title'],
                "part_title": chapter_info['part_title'],
                "chapter_metadata": {
                    "cluster_count": len(chapter_info['clusters']),
                },
                "clusters": []
            }
            
            chapter_quality_scores = []
            chapter_reading_time = 0
            chapter_audio_length = 0
            chapter_voice_memo_count = 0
            chapter_creation_dates = []
            
            # Process clusters in chapter
            for cluster_id in chapter_info['clusters']:
                cluster_id_str = str(cluster_id)
                
                # Get cluster description
                cluster_desc = cluster_desc_lookup.get(cluster_id_str, {})
                cluster_name = cluster_desc.get('title', f'Cluster {cluster_id}')
                cluster_description = cluster_desc.get('raw_essence', 'No description available')
                
                # Find ALL voice memos for this cluster (from unfiltered data)
                cluster_voice_memos = []
                
                # Get voice memo filenames from cluster
                if cluster_id_str in all_enriched_data.get('clusters', {}):
                    for vm in all_enriched_data['clusters'][cluster_id_str].get('voice_memos', []):
                        if vm['filename'] in all_voice_memos:
                            cluster_voice_memos.append(vm['filename'])
                
                if not cluster_voice_memos:
                    continue
                
                # Process voice memos with rich metadata
                enriched_voice_memos = []
                cluster_quality_scores = []
                cluster_reading_time = 0
                cluster_audio_length = 0
                
                for filename in cluster_voice_memos:
                    voice_memo = all_voice_memos[filename]
                    
                    # Load full transcript and fingerprint
                    transcript = load_transcript(filename)
                    fingerprint = load_fingerprint(filename)
                    
                    if not transcript and not fingerprint:
                        continue
                    
                    # Calculate text metadata
                    text_metadata = calculate_text_metadata(transcript or "")
                    
                    # Get file sizes
                    transcript_path = Path("data/00 transcripts") / filename.replace('.json', '.txt')
                    fingerprint_path = Path("data/01 fingerprints") / filename
                    
                    # Get audio metadata  
                    audio_filename = filename.replace('.json', '.m4a')
                    audio_path = Path("audio_files") / audio_filename
                    audio_metadata = get_audio_metadata(audio_path)
                    
                    # Get quality coefficient for grading
                    quality_coefficient = voice_memo.get('quality_coefficient', 0.0)
                    if fingerprint and 'insight_quality' in fingerprint:
                        quality_coefficient = fingerprint['insight_quality'].get('quality_coefficient', 0.0)
                    
                    # Use curved A-F grading for voice memos
                    quality_grade = voice_memo_grade_mapping.get(quality_coefficient, "B+")
                    
                    # Combine all metadata
                    voice_memo_metadata = {
                        "file_size_bytes": (
                            get_file_size(transcript_path) + 
                            get_file_size(fingerprint_path) + 
                            audio_metadata["file_size_bytes"]
                        ),
                        "word_count": text_metadata["word_count"],
                        "estimated_reading_time_seconds": text_metadata["estimated_reading_time_seconds"],
                        "estimated_reading_time_pretty": format_duration_pretty(text_metadata["estimated_reading_time_seconds"]),
                        "audio_length_seconds": audio_metadata["audio_length_seconds"],
                        "audio_length_pretty": format_duration_pretty(audio_metadata["audio_length_seconds"]),
                        "creation_date": audio_metadata["creation_date"],
                        "creation_date_pretty": format_date_pretty(audio_metadata["creation_date"]) if audio_metadata["creation_date"] else None,
                        "quality_coefficient": quality_coefficient,
                        "quality_grade": quality_grade
                    }
                    
                    enriched_voice_memo = {
                        "filename": filename,
                        "transcript": transcript or "",
                        "fingerprint": fingerprint or {},
                        "metadata": voice_memo_metadata
                    }
                    
                    enriched_voice_memos.append(enriched_voice_memo)
                    
                    # Aggregate for cluster
                    if quality_coefficient:
                        cluster_quality_scores.append(quality_coefficient)
                    cluster_reading_time += voice_memo_metadata["estimated_reading_time_seconds"]
                    cluster_audio_length += voice_memo_metadata["audio_length_seconds"]
                    if voice_memo_metadata["creation_date"]:
                        chapter_creation_dates.append(voice_memo_metadata["creation_date"])
                
                # Calculate temporal metadata for cluster
                cluster_creation_dates = [vm["metadata"]["creation_date"] for vm in enriched_voice_memos if vm["metadata"]["creation_date"]]
                cluster_temporal_metadata = calculate_temporal_metadata(cluster_creation_dates)
                
                # Calculate cluster average quality (store for curved grading later)
                avg_quality = sum(cluster_quality_scores) / len(cluster_quality_scores) if cluster_quality_scores else 0
                
                # Build cluster data
                cluster_output = {
                    "cluster_id": cluster_id_str,
                    "cluster_name": cluster_name,
                    "cluster_description": cluster_description,
                    "cluster_metadata": {
                        "voice_memo_count": len(enriched_voice_memos),
                        "average_quality_coefficient": avg_quality,
                        "quality_grade": cluster_quality_grade,
                        "total_reading_time_minutes": round(cluster_reading_time / 60, 2),
                        "total_reading_time_pretty": format_duration_pretty(cluster_reading_time),
                        "total_audio_length_seconds": round(cluster_audio_length, 1),
                        "total_audio_length_pretty": format_duration_pretty(cluster_audio_length),
                        **cluster_temporal_metadata
                    },
                    "voice_memos": enriched_voice_memos
                }
                
                chapter_data["clusters"].append(cluster_output)
                
                # Aggregate for chapter
                chapter_quality_scores.extend(cluster_quality_scores)
                chapter_reading_time += cluster_reading_time
                chapter_audio_length += cluster_audio_length
                chapter_voice_memo_count += len(enriched_voice_memos)
            
            # Calculate chapter temporal metadata and average quality (store for curved grading later)
            chapter_temporal_metadata = calculate_temporal_metadata(chapter_creation_dates)
            chapter_avg_quality = sum(chapter_quality_scores) / len(chapter_quality_scores) if chapter_quality_scores else 0
            
            # Update chapter metadata
            chapter_data["chapter_metadata"].update({
                "voice_memo_count": chapter_voice_memo_count,
                "average_quality_coefficient": chapter_avg_quality,
                "quality_grade": chapter_quality_grade,
                "total_reading_time_minutes": round(chapter_reading_time / 60, 2),
                "total_reading_time_pretty": format_duration_pretty(chapter_reading_time),
                "total_audio_length_seconds": round(chapter_audio_length, 1),
                "total_audio_length_pretty": format_duration_pretty(chapter_audio_length),
                **chapter_temporal_metadata
            })
            
            book_data["chapters"].append(chapter_data)
            
            # Aggregate for book
            book_data["book_metadata"]["total_clusters"] += len(chapter_data["clusters"])
            book_data["book_metadata"]["total_voice_memos"] += chapter_voice_memo_count
            book_data["book_metadata"]["total_reading_time_minutes"] += chapter_data["chapter_metadata"]["total_reading_time_minutes"]
            book_data["book_metadata"]["total_audio_length_seconds"] += chapter_data["chapter_metadata"]["total_audio_length_seconds"]
            book_quality_scores.extend(chapter_quality_scores)
            book_creation_dates.extend(chapter_creation_dates)
        
        # Calculate final book-level metadata with composite scoring
        book_temporal_metadata = calculate_temporal_metadata(book_creation_dates)
        book_avg_quality = sum(book_quality_scores) / len(book_quality_scores) if book_quality_scores else 0
        
        # Calculate composite book score with all sub-scores
        all_voice_memos = []
        for chapter in book_data["chapters"]:
            for cluster in chapter["clusters"]:
                for vm in cluster["voice_memos"]:
                    all_voice_memos.append(vm["metadata"])
        
        composite_scoring = calculate_composite_book_score(
            all_voice_memos, 
            book_data["book_metadata"]["total_voice_memos"]
        )
        
        # Update book metadata with comprehensive scoring
        book_data["book_metadata"].update({
            "average_quality_coefficient": book_avg_quality,
            "total_reading_time_pretty": format_duration_pretty(book_data["book_metadata"]["total_reading_time_minutes"] * 60),
            "total_audio_length_pretty": format_duration_pretty(book_data["book_metadata"]["total_audio_length_seconds"]),
            # Composite scoring components
            "volume_factor": composite_scoring["volume_factor"],
            "excellence_count": composite_scoring["excellence_count"],
            "excellence_percentage": composite_scoring["excellence_pct"],
            "high_quality_count": composite_scoring["high_quality_count"],
            "high_quality_percentage": composite_scoring["high_quality_pct"],
            "average_quality_factor": composite_scoring["avg_quality_factor"],
            "composite_score": composite_scoring["composite_score"],
            # Pillar taxonomy and recommendations
            "pillar_stage": composite_scoring["pillar_stage"],
            "stage_description": composite_scoring["stage_description"],
            "recommended_action": composite_scoring["recommended_action"],
            **book_temporal_metadata
        })
        
        # Save book file
        book_filename = f"{book_title}.json"
        book_output_path = output_dir / book_filename
        
        with open(book_output_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Exported: {book_filename}")
        print(f"   📊 {book_data['book_metadata']['total_chapters']} chapters, "
              f"{book_data['book_metadata']['total_clusters']} clusters, "
              f"{book_data['book_metadata']['total_voice_memos']} voice memos")
        
        books_processed += 1
    
    print(f"\n🎉 Full library export complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Summary: {books_processed} books exported with ALL voice memos")
    print(f"📈 Grading: A-F scale applied to all {len(all_voice_memos)} records")
    
    return 0

if __name__ == "__main__":
    import sys
    
    # Check what operation is requested
    if len(sys.argv) > 1:
        operation = sys.argv[1].lower()
        if operation == "filter":
            exit(filter_clusters_by_quality())
        elif operation == "books":
            exit(create_comprehensive_enriched_clusters())
        elif operation == "library":
            print("⚠️  Legacy 'library' command deprecated. Use 'full' for enhanced export.")
            exit(create_enhanced_full_library_export())
        elif operation == "full":
            exit(create_enhanced_full_library_export())
        else:
            print("Usage: python enrich_semantic_clusters.py [filter|books|library|full]")
            print("  (no args): Run enrichment (export medium and coarse clusters)")
            print("  filter: Filter enriched clusters by quality ≥ 0.6")
            print("  books: Create comprehensive book structure with descriptions")
            print("  library: Legacy command (redirects to 'full')")
            print("  full: Enhanced export with curved A-F grading + composite book scoring")
            exit(1)
    else:
        exit(main())