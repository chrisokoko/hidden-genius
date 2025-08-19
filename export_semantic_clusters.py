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

def load_clustering_results(file_path: str = "data/clusters/intelligent_hierarchical_results.json") -> Optional[Dict[str, Any]]:
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

def load_transcript(file_path: str, transcripts_base: str = "data/transcripts") -> Optional[str]:
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

def load_fingerprint(file_path: str, fingerprints_base: str = "data/fingerprints") -> Optional[Dict[str, Any]]:
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
                    memo_data = {
                        "filename": file_path,
                        "transcript": transcript,
                        "fingerprint": fingerprint
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

def main():
    """Main export process"""
    print("🗂️  SEMANTIC CLUSTER EXPORT")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data/semantic_clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading clustering results...")
    clustering_results = load_clustering_results()
    if not clustering_results:
        return 1
    
    print("📂 Loading cluster metadata...")
    cluster_metadata = load_cluster_metadata()
    
    # Build hierarchical mapping
    print("🗺️  Building hierarchical mapping...")
    mapping_data = build_hierarchical_mapping(clustering_results)
    
    # Export each coarse cluster
    coarse_clusters = mapping_data['hierarchy']['coarse']
    print(f"📤 Exporting {len(coarse_clusters)} coarse clusters...")
    
    exported_files = []
    for coarse_id, coarse_files in coarse_clusters.items():
        print(f"\n🎯 Processing Coarse Cluster {coarse_id} ({len(coarse_files)} voice memos)...")
        
        # Export cluster data
        cluster_data = export_coarse_cluster(coarse_id, coarse_files, mapping_data, cluster_metadata)
        
        # Generate filename
        cluster_title = cluster_data['cluster_metadata']['title']
        filename = sanitize_filename(cluster_title) + '.json'
        output_path = output_dir / filename
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)
        
        exported_files.append({
            'cluster_id': coarse_id,
            'title': cluster_title,
            'filename': filename,
            'voice_memos': cluster_data['cluster_metadata']['statistics']['total_voice_memos'],
            'medium_clusters': cluster_data['cluster_metadata']['statistics']['medium_clusters'],
            'fine_clusters': cluster_data['cluster_metadata']['statistics']['fine_clusters']
        })
        
        print(f"   ✅ Saved: {filename}")
        print(f"   📊 Stats: {cluster_data['cluster_metadata']['statistics']['total_voice_memos']} memos, " +
              f"{cluster_data['cluster_metadata']['statistics']['medium_clusters']} medium, " +
              f"{cluster_data['cluster_metadata']['statistics']['fine_clusters']} fine clusters")
    
    # Create summary file
    summary_path = output_dir / "export_summary.json"
    summary_data = {
        "exported_at": datetime.now().isoformat(),
        "total_coarse_clusters": len(exported_files),
        "total_voice_memos": sum(f['voice_memos'] for f in exported_files),
        "exported_files": exported_files
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Export complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📊 Summary:")
    print(f"   • {len(exported_files)} coarse clusters exported")
    print(f"   • {summary_data['total_voice_memos']} total voice memos")
    print(f"   • Average {summary_data['total_voice_memos'] // len(exported_files)} memos per cluster")
    
    return 0

if __name__ == "__main__":
    exit(main())