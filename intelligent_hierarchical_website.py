#!/usr/bin/env python3
"""
Intelligent Hierarchical Clustering Website Generator

Creates an interactive HTML website for exploring the intelligent hierarchical clustering results.
Shows the three semantic levels (major themes, sub-themes, specific topics) with quality metrics
and allows drilling down into each cluster.

Usage:
    python intelligent_hierarchical_website.py

This will:
1. Load intelligent hierarchical clustering results from data/intelligent_hierarchical_results.json
2. Load semantic fingerprints from data/fingerprints/ directory 
3. Generate an interactive HTML website for exploring the semantic hierarchy
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_intelligent_results(file_path: str = "data/clusters/intelligent_hierarchical_results.json") -> Optional[Dict[str, Any]]:
    """Load intelligent hierarchical clustering results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("   Make sure you've run 'python mega_script.py cluster' first.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file {file_path}: {e}")
        return None


def load_cluster_metadata(file_path: str = "data/clusters/cluster_metadata.json") -> Optional[Dict[str, Any]]:
    """Load cluster titles and descriptions from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Cluster metadata not found: {file_path}")
        print("   Website will show generic cluster names")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in cluster metadata {file_path}: {e}")
        return None


def build_hierarchical_tree(intelligent_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a hierarchical tree structure showing coarse -> medium -> fine relationships"""
    
    # Get the core data
    file_mapping = intelligent_results.get('file_mapping', [])
    optimal_levels = intelligent_results.get('optimal_levels', {})
    
    coarse_labels = optimal_levels.get('coarse', {}).get('labels', [])
    medium_labels = optimal_levels.get('medium', {}).get('labels', [])
    fine_labels = optimal_levels.get('fine', {}).get('labels', [])
    
    coarse_clusters = optimal_levels.get('coarse', {}).get('clusters', {})
    medium_clusters = optimal_levels.get('medium', {}).get('clusters', {})
    fine_clusters = optimal_levels.get('fine', {}).get('clusters', {})
    
    # Build mapping from file to index
    file_to_index = {file_path: idx for idx, file_path in enumerate(file_mapping)}
    
    # Build the tree structure
    tree = {}
    
    # Process each coarse cluster
    for coarse_id, coarse_files in coarse_clusters.items():
        coarse_id = str(coarse_id)
        tree[coarse_id] = {
            'files': coarse_files,
            'medium_clusters': {}
        }
        
        # Find which medium clusters belong to this coarse cluster
        medium_ids_in_coarse = set()
        for file_path in coarse_files:
            if file_path in file_to_index:
                file_idx = file_to_index[file_path]
                if file_idx < len(medium_labels):
                    medium_ids_in_coarse.add(str(medium_labels[file_idx]))
        
        # Process each medium cluster within this coarse cluster
        for medium_id in medium_ids_in_coarse:
            if medium_id in medium_clusters:
                medium_files = medium_clusters[medium_id]
                tree[coarse_id]['medium_clusters'][medium_id] = {
                    'files': medium_files,
                    'fine_clusters': {}
                }
                
                # Find which fine clusters belong to this medium cluster
                fine_ids_in_medium = set()
                for file_path in medium_files:
                    if file_path in file_to_index:
                        file_idx = file_to_index[file_path]
                        if file_idx < len(fine_labels):
                            fine_ids_in_medium.add(str(fine_labels[file_idx]))
                
                # Process each fine cluster within this medium cluster
                for fine_id in fine_ids_in_medium:
                    if fine_id in fine_clusters:
                        fine_files = fine_clusters[fine_id]
                        tree[coarse_id]['medium_clusters'][medium_id]['fine_clusters'][fine_id] = {
                            'files': fine_files
                        }
    
    return tree


def load_semantic_fingerprints(fingerprints_dir: str = "data/fingerprints") -> Dict[str, Any]:
    """Load all semantic fingerprints from the fingerprints directory"""
    fingerprints = {}
    fingerprints_path = Path(fingerprints_dir)
    
    if not fingerprints_path.exists():
        print(f"⚠️  Fingerprints directory not found: {fingerprints_dir}")
        print("   Website will show filenames only")
        return fingerprints
    
    json_files = list(fingerprints_path.rglob("*.json"))
    print(f"📂 Loading semantic fingerprints from {len(json_files)} files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                fingerprint_data = json.load(f)
            
            relative_path = json_file.relative_to(fingerprints_path)
            original_filename = str(relative_path).replace('.json', '')
            fingerprints[original_filename] = fingerprint_data
            
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    print(f"✅ Loaded {len(fingerprints)} semantic fingerprints")
    return fingerprints


def extract_cluster_insights(cluster_filenames: List[str], semantic_fingerprints: Dict[str, Any]) -> Dict[str, Any]:
    """Extract unified insight modules for a cluster"""
    insights = {
        'unified_insights': [],
        'missing_files': [],
        'cluster_themes': [],
        'quality_indicators': {}
    }
    
    # Extract themes for cluster characterization
    themes = []
    central_questions = []
    breakthrough_moments = []
    
    for filename in cluster_filenames:
        base_filename = filename.replace('.json', '')
        fingerprint = None
        
        possible_keys = [filename, base_filename, f"{base_filename}.json"]
        for key in possible_keys:
            if key in semantic_fingerprints:
                fingerprint = semantic_fingerprints[key]
                break
        
        if fingerprint:
            # Extract core exploration elements
            core_exploration = fingerprint.get('core_exploration', {})
            central_question = core_exploration.get('central_question')
            breakthrough_moment = core_exploration.get('breakthrough_moment')
            
            if central_question:
                central_questions.append(central_question)
            if breakthrough_moment:
                breakthrough_moments.append(breakthrough_moment)
            
            # Extract conceptual DNA
            conceptual_dna = fingerprint.get('conceptual_dna', [])
            themes.extend(conceptual_dna)
            
            raw_essence = fingerprint.get('raw_essence')
            
            insights['unified_insights'].append({
                'question': central_question,
                'essence': raw_essence,
                'breakthrough': breakthrough_moment,
                'filename': filename,
                'conceptual_dna': conceptual_dna,
                'full_semantic_fingerprint': fingerprint
            })
        else:
            insights['missing_files'].append(filename)
    
    # Generate cluster characterization
    insights['cluster_themes'] = list(set(themes))[:5]  # Top 5 unique themes
    insights['central_questions'] = list(set(central_questions))[:3]  # Top 3 questions
    insights['breakthrough_moments'] = list(set(breakthrough_moments))[:3]  # Top 3 breakthroughs
    
    # Quality indicators
    insights['quality_indicators'] = {
        'total_items': len(cluster_filenames),
        'items_with_fingerprints': len(insights['unified_insights']),
        'coverage_ratio': len(insights['unified_insights']) / len(cluster_filenames) if cluster_filenames else 0,
        'theme_diversity': len(insights['cluster_themes'])
    }
    
    return insights


def generate_cluster_html(cluster_id: int, cluster_data: Dict[str, Any], insights: Dict[str, Any], level_name: str, cluster_metadata: Optional[Dict[str, Any]] = None, outlier_class: str = "") -> str:
    """Generate HTML for a single cluster"""
    filenames = cluster_data.get('filenames', [])
    quality = insights['quality_indicators']
    
    # Get cluster title and description from metadata
    cluster_title = f"Cluster {cluster_id + 1}"  # Default fallback
    cluster_description = None
    
    if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
        level_metadata = cluster_metadata['cluster_titles_descriptions'].get(level_name, {})
        cluster_info = level_metadata.get(str(cluster_id), {})
        if cluster_info:
            cluster_title = cluster_info.get('title', cluster_title)
            cluster_description = cluster_info.get('description')
    
    cluster_html = f"""
    <div class="cluster-card{outlier_class}" id="{level_name}_cluster_{cluster_id}">
        <div class="cluster-header">
            <h4>{cluster_title}</h4>
            <div class="cluster-stats">
                <span class="stat">{quality['total_items']} items</span>
                <span class="stat">{quality['coverage_ratio']:.1%} coverage</span>
                <span class="stat">{quality['theme_diversity']} themes</span>
            </div>
        </div>"""
    
    # Add cluster definition if available
    if cluster_description:
        cluster_html += f"""
        
        <div class="cluster-definition">
            <h5>Cluster Definition:</h5>
            <p class="definition-text">{cluster_description}</p>
        </div>"""
    
    cluster_html += """
        
        <div class="cluster-items">
            <details>
                <summary>View Items ({}) →</summary>
                <div class="items-grid">
    """.format(len(filenames))
    
    for insight in insights['unified_insights']:
        item_html = f"""
                    <div class="item-card">
                        <div class="item-filename">{insight['filename']}</div>
        """
        
        if insight['question']:
            item_html += f'<div class="item-question">Q: {insight["question"]}</div>'
        
        if insight['essence']:
            item_html += f'<div class="item-essence">{insight["essence"]}</div>'
        
        if insight['breakthrough']:
            item_html += f'<div class="item-breakthrough">💡 {insight["breakthrough"]}</div>'
        
        item_html += """
                    </div>
        """
        cluster_html += item_html
    
    # Show missing files
    for filename in insights['missing_files']:
        cluster_html += f"""
                    <div class="item-card missing">
                        <div class="item-filename">{filename}</div>
                        <div class="item-note">No semantic fingerprint available</div>
                    </div>
        """
    
    cluster_html += """
                </div>
            </details>
        </div>
    </div>
    """
    
    return cluster_html


def generate_level_html(level_name: str, level_data: Dict[str, Any], semantic_fingerprints: Dict[str, Any], cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate HTML for a complete level (major/sub/specific)"""
    clusters = level_data.get('clusters', {})
    n_clusters = level_data.get('n_clusters', 0)
    quality_score = level_data.get('quality_score', 0)
    silhouette = level_data.get('silhouette', 0)
    height = level_data.get('height', 0)
    
    # Calculate cluster size breakdown
    cluster_sizes = [len(filenames) for filenames in clusters.values()]
    one_item_clusters = sum(1 for size in cluster_sizes if size == 1)
    two_item_clusters = sum(1 for size in cluster_sizes if size == 2)
    multi_item_clusters = sum(1 for size in cluster_sizes if size > 2)
    
    level_html = f"""
    <div class="level-section" id="{level_name}_level">
        <div class="level-header">
            <h2>{level_name.title()} Themes</h2>
            <div class="level-metrics">
                <div class="metric">
                    <span class="metric-label">Cut Height:</span>
                    <span class="metric-value">{height:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Clusters:</span>
                    <span class="metric-value">{n_clusters}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality:</span>
                    <span class="metric-value">{quality_score:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Silhouette:</span>
                    <span class="metric-value">{silhouette:.3f}</span>
                </div>
            </div>
            
            <div class="cluster-breakdown">
                <div class="breakdown-stats">
                    <span class="breakdown-label">Find Themes:</span>
                    <span class="breakdown-item">1-item: {one_item_clusters}</span>
                    <span class="breakdown-item">2-item: {two_item_clusters}</span>
                    <span class="breakdown-item">Multi-item: {multi_item_clusters}</span>
                </div>
                <div class="outlier-controls">
                    <label class="toggle-switch">
                        <input type="checkbox" id="{level_name}_outlier_toggle" onchange="toggleOutliers('{level_name}')">
                        <span class="slider"></span>
                        Show Outliers (1-item clusters)
                    </label>
                </div>
            </div>
        </div>
        
        <div class="clusters-container" id="{level_name}_clusters_container">
    """
    
    # Sort clusters by size (number of items) in descending order
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Process each cluster
    for cluster_id, cluster_filenames in sorted_clusters:
        if isinstance(cluster_id, str):
            cluster_id = int(cluster_id)
        
        # Mark outliers (1-item clusters) for toggle functionality
        is_outlier = len(cluster_filenames) == 1
        outlier_class = " outlier-cluster" if is_outlier else ""
        
        cluster_data = {'filenames': cluster_filenames, 'is_outlier': is_outlier}
        insights = extract_cluster_insights(cluster_filenames, semantic_fingerprints)
        cluster_html = generate_cluster_html(cluster_id, cluster_data, insights, level_name, cluster_metadata, outlier_class)
        level_html += cluster_html
    
    level_html += """
        </div>
    </div>
    """
    
    return level_html


def generate_tree_html_content(intelligent_results: Dict[str, Any], semantic_fingerprints: Dict[str, Any], cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate the complete HTML content for the hierarchical tree website"""
    
    # Build the hierarchical tree structure
    tree = build_hierarchical_tree(intelligent_results)
    summary = intelligent_results.get('summary', {})
    
    all_evaluations = intelligent_results.get('all_evaluations', {})
    summary = intelligent_results.get('summary', {})
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Hierarchical Clustering Explorer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .header h1 {{
            font-size: 2.8rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 20px;
        }}
        
        .summary-stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }}
        
        .summary-stat {{
            text-align: center;
        }}
        
        .summary-stat .number {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .summary-stat .label {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
        }}
        
        .navigation {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .nav-buttons {{
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .nav-button {{
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            text-decoration: none;
        }}
        
        .nav-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .nav-button.active {{
            background: linear-gradient(135deg, #11998e, #38ef7d);
        }}
        
        .level-section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            display: none;
        }}
        
        .level-section.active {{
            display: block;
        }}
        
        .level-header {{
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .level-header h2 {{
            font-size: 2.2rem;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .level-metrics {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        
        .metric {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px 20px;
            background: #f8f9fa;
            border-radius: 12px;
            min-width: 120px;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: #666;
            text-transform: uppercase;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.4rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .clusters-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .cluster-card {{
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }}
        
        .cluster-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}
        
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .cluster-header h4 {{
            font-size: 1.3rem;
            color: #333;
            margin: 0;
        }}
        
        .cluster-stats {{
            display: flex;
            gap: 10px;
        }}
        
        .stat {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        .cluster-themes {{
            margin-bottom: 15px;
        }}
        
        .cluster-themes h5 {{
            font-size: 1rem;
            margin-bottom: 8px;
            color: #555;
        }}
        
        .theme-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        
        .theme-tag {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        .cluster-definition {{
            margin-bottom: 15px;
        }}
        
        .cluster-definition h5 {{
            font-size: 1rem;
            margin-bottom: 8px;
            color: #555;
        }}
        
        .definition-text {{
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
            font-size: 0.9rem;
            color: #444;
            line-height: 1.5;
            margin: 0;
        }}
        
        .cluster-questions {{
            margin-bottom: 15px;
        }}
        
        .cluster-questions h5 {{
            font-size: 1rem;
            margin-bottom: 8px;
            color: #555;
        }}
        
        .cluster-questions ul {{
            list-style: none;
            padding: 0;
        }}
        
        .cluster-questions li {{
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 5px;
            border-left: 3px solid #667eea;
            font-size: 0.9rem;
        }}
        
        .cluster-items details {{
            margin-top: 10px;
        }}
        
        .cluster-items summary {{
            cursor: pointer;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            font-weight: 500;
            color: #667eea;
            border: 1px solid #e9ecef;
        }}
        
        .cluster-items summary:hover {{
            background: #e9ecef;
        }}
        
        .items-grid {{
            display: grid;
            gap: 10px;
            margin-top: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .item-card {{
            background: #fafafa;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        
        .item-card.missing {{
            background: #fff3e0;
            border-color: #ffb74d;
        }}
        
        .item-filename {{
            font-weight: 600;
            color: #333;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }}
        
        .item-question {{
            font-style: italic;
            color: #667eea;
            font-size: 0.85rem;
            margin-bottom: 5px;
        }}
        
        .item-essence {{
            color: #555;
            font-size: 0.85rem;
            margin-bottom: 5px;
        }}
        
        .item-breakthrough {{
            background: #e8f5e8;
            color: #2e7d32;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-top: 5px;
        }}
        
        .item-note {{
            color: #f57f17;
            font-size: 0.8rem;
            font-style: italic;
        }}
        
        .evaluations-section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            display: none;
        }}
        
        .evaluations-section.active {{
            display: block;
        }}
        
        .evaluations-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .evaluations-table th,
        .evaluations-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .evaluations-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        
        .evaluations-table tr:hover {{
            background: #f8f9fa;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .summary-stats {{
                flex-direction: column;
                gap: 15px;
            }}
            
            .nav-buttons {{
                flex-direction: column;
            }}
            
            .level-metrics {{
                flex-direction: column;
                gap: 15px;
            }}
            
            .clusters-container {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* New styles for cluster breakdown and outlier toggle */
        .cluster-breakdown {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .breakdown-stats {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .breakdown-label {{
            font-weight: 600;
            color: #333;
        }}
        
        .breakdown-item {{
            padding: 4px 8px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            font-size: 0.9em;
            color: #333;
        }}
        
        .outlier-controls {{
            display: flex;
            align-items: center;
        }}
        
        .toggle-switch {{
            position: relative;
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            font-size: 0.9em;
            color: #333;
            gap: 10px;
        }}
        
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        
        .slider {{
            position: relative;
            width: 50px;
            height: 24px;
            background: #ccc;
            border-radius: 24px;
            transition: background 0.3s;
        }}
        
        .slider:before {{
            content: "";
            position: absolute;
            width: 20px;
            height: 20px;
            left: 2px;
            top: 2px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }}
        
        .toggle-switch input:checked + .slider {{
            background: #667eea;
        }}
        
        .toggle-switch input:checked + .slider:before {{
            transform: translateX(26px);
        }}
        
        /* Outlier cluster styling */
        .outlier-cluster {{
            opacity: 0.6;
            border-left: 3px solid #ff6b6b;
        }}
        
        .outlier-cluster.hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Intelligent Hierarchical Clustering</h1>
            <div class="subtitle">Three-Phase Semantic Analysis: Natural Breakpoints → Quality Validation → Level Assignment</div>
            
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="number">{summary.get('total_analyzed', summary.get('total_embeddings', 0))}</div>
                    <div class="label">Total Analyzed</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{summary.get('total_with_homes', summary.get('total_analyzed', 0))}</div>
                    <div class="label">Found Homes</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{summary.get('total_outliers', 0)}</div>
                    <div class="label">Outliers</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{len(optimal_levels)}</div>
                    <div class="label">Semantic Levels</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{len(all_evaluations)}</div>
                    <div class="label">Heights Evaluated</div>
                </div>
            </div>
        </div>
        
        <div class="navigation">
            <div class="nav-buttons">
    """
    
    # Generate dynamic navigation buttons
    level_icons = {'coarse': '🎯', 'medium': '📋', 'fine': '🔍', 'major': '🎯', 'sub': '📋', 'specific': '🔍'}
    first_level = True
    
    for level_name in optimal_levels.keys():
        icon = level_icons.get(level_name, '📊')
        active_class = " active" if first_level else ""
        level_title = level_name.title() + " Level"
        html_content += f'                <button class="nav-button{active_class}" onclick="showLevel(\'{level_name}\')">{icon} {level_title}</button>\n'
        first_level = False
    
    html_content += '                <button class="nav-button" onclick="showLevel(\'evaluations\')">📊 Quality Evaluations</button>\n'
    html_content += """            </div>
        </div>
        
        <!-- Level sections will be inserted here -->
    """
    
    # Add each level (dynamically discovered)
    for level_name, level_data in optimal_levels.items():
        level_html = generate_level_html(level_name, level_data, semantic_fingerprints, cluster_metadata)
        html_content += level_html
    
    # Add evaluations section
    html_content += f"""
        <div class="evaluations-section" id="evaluations_level">
            <div class="level-header">
                <h2>Quality Evaluations</h2>
                <div class="level-metrics">
                    <div class="metric">
                        <span class="metric-label">Heights Tested:</span>
                        <span class="metric-value">{len(all_evaluations)}</span>
                    </div>
                </div>
            </div>
            
            <table class="evaluations-table">
                <thead>
                    <tr>
                        <th>Cut Height</th>
                        <th>Clusters</th>
                        <th>Quality Score</th>
                        <th>Silhouette</th>
                        <th>Balance</th>
                        <th>Separation</th>
                        <th>Used In Level</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Sort evaluations by cut height (ascending)
    sorted_evaluations = sorted(all_evaluations.items(), key=lambda x: float(x[0]))
    
    for height_str, eval_data in sorted_evaluations:
        height = float(height_str)
        
        # Check if this height is used in any level
        used_in = []
        for level_name, level_data in optimal_levels.items():
            if level_data and abs(level_data['height'] - height) < 1e-6:
                used_in.append(level_name)
        
        used_in_str = ', '.join(used_in) if used_in else '—'
        
        html_content += f"""
                    <tr>
                        <td>{height:.3f}</td>
                        <td>{eval_data.get('n_clusters', 0)}</td>
                        <td>{eval_data.get('quality_score', 0):.3f}</td>
                        <td>{eval_data.get('silhouette', 0):.3f}</td>
                        <td>{eval_data.get('balance_score', 0):.3f}</td>
                        <td>{eval_data.get('separation_ratio', 0):.2f}</td>
                        <td><strong>{used_in_str}</strong></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    """
    
    # Add JavaScript
    html_content += """
    </div>
    
    <script>
        function showLevel(levelName) {
            // Hide all sections
            document.querySelectorAll('.level-section, .evaluations-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from all buttons
            document.querySelectorAll('.nav-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected section
            const targetSection = document.getElementById(levelName + '_level');
            if (targetSection) {
                targetSection.classList.add('active');
            }
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        // Show first level by default
        document.addEventListener('DOMContentLoaded', function() {
            const firstLevelSection = document.querySelector('.level-section');
            if (firstLevelSection) {
                firstLevelSection.classList.add('active');
            }
        });
        
        // Toggle outliers (1-item clusters) visibility
        function toggleOutliers(levelName) {
            const toggle = document.getElementById(levelName + '_outlier_toggle');
            const container = document.getElementById(levelName + '_clusters_container');
            const outlierClusters = container.querySelectorAll('.outlier-cluster');
            
            if (toggle.checked) {
                // Show outliers
                outlierClusters.forEach(cluster => {
                    cluster.classList.remove('hidden');
                });
            } else {
                // Hide outliers
                outlierClusters.forEach(cluster => {
                    cluster.classList.add('hidden');
                });
            }
        }
        
        // Initialize outliers as hidden by default
        document.addEventListener('DOMContentLoaded', function() {
            const allOutliers = document.querySelectorAll('.outlier-cluster');
            allOutliers.forEach(cluster => {
                cluster.classList.add('hidden');
            });
        });
    </script>
</body>
</html>
    """
    
    return html_content


def main():
    """Main function to generate the intelligent hierarchical website"""
    print("🧠 Generating Intelligent Hierarchical Clustering Website")
    print("=" * 60)
    
    # Load intelligent hierarchical results
    print("📂 Loading intelligent hierarchical clustering results...")
    intelligent_results = load_intelligent_results()
    if not intelligent_results:
        return 1
    
    # Load semantic fingerprints
    print("📂 Loading semantic fingerprints...")
    semantic_fingerprints = load_semantic_fingerprints()
    
    # Load cluster metadata
    print("📂 Loading cluster metadata...")
    cluster_metadata = load_cluster_metadata()
    
    # Generate HTML content
    print("🔨 Generating HTML content...")
    html_content = generate_html_content(intelligent_results, semantic_fingerprints, cluster_metadata)
    
    # Save the website
    output_file = Path("data/websites/intelligent_hierarchical_website.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Website generated successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"🌐 Open in browser to explore your intelligent hierarchical clustering")
    
    # Show summary with dynamic level names
    optimal_levels = intelligent_results.get('optimal_levels', {})
    print(f"\n📊 Summary:")
    if optimal_levels:
        for level_name, level_data in optimal_levels.items():
            if level_data:
                print(f"  {level_name.title()}: {level_data['n_clusters']} clusters, quality={level_data['quality_score']:.3f}")
            else:
                print(f"  {level_name.title()}: No suitable clustering found")
    else:
        print("  No optimal levels found")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())