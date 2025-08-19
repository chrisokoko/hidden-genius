#!/usr/bin/env python3
"""
Hierarchical Tree Website Generator

Creates an interactive HTML website showing the true hierarchical tree structure:
- Coarse clusters at the top level
- Medium clusters nested within coarse clusters  
- Fine clusters nested within medium clusters
- Voice memo files at the fine level
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


def load_cluster_metadata(cluster_names_dir: str = "data/cluster_names") -> Optional[Dict[str, Any]]:
    """Load cluster titles and descriptions from batch files in cluster_names directory"""
    try:
        cluster_names_path = Path(cluster_names_dir)
        if not cluster_names_path.exists():
            print(f"⚠️  Cluster names directory not found: {cluster_names_dir}")
            print("   Website will show generic cluster names")
            return None
        
        # Find all batch files
        batch_files = list(cluster_names_path.glob("*_batch_*.json"))
        if not batch_files:
            print(f"⚠️  No batch files found in: {cluster_names_dir}")
            print("   Website will show generic cluster names")
            return None
        
        print(f"📂 Loading cluster metadata from {len(batch_files)} batch files...")
        
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
                            
                print(f"   ✅ Loaded: {batch_file.name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {batch_file.name}: {e}")
        
        # Count total clusters loaded
        total_clusters = sum(
            len(combined_metadata['cluster_titles_descriptions'][level])
            for level in ['coarse', 'medium', 'fine']
        )
        print(f"📊 Total clusters loaded: {total_clusters}")
        
        return combined_metadata
        
    except Exception as e:
        print(f"❌ Error loading cluster metadata: {e}")
        return None


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


def get_cluster_info(cluster_id: str, level: str, cluster_metadata: Optional[Dict[str, Any]]) -> tuple:
    """Get title and description for a cluster from metadata"""
    title = f"Cluster {cluster_id}"
    description = None
    
    if cluster_metadata and 'cluster_titles_descriptions' in cluster_metadata:
        level_metadata = cluster_metadata['cluster_titles_descriptions'].get(level, {})
        cluster_info = level_metadata.get(cluster_id, {})
        if cluster_info:
            title = cluster_info.get('title', title)
            description = cluster_info.get('description')
    
    return title, description


def generate_file_html(file_path: str, semantic_fingerprints: Dict[str, Any]) -> str:
    """Generate HTML for a single voice memo file"""
    base_filename = file_path.replace('.json', '')
    fingerprint = None
    
    # Try to find the fingerprint
    possible_keys = [file_path, base_filename, f"{base_filename}.json"]
    for key in possible_keys:
        if key in semantic_fingerprints:
            fingerprint = semantic_fingerprints[key]
            break
    
    # Extract insights
    filename_display = base_filename.split('/')[-1]  # Just the filename
    
    html = f'<div class="file-item">'
    html += f'<div class="file-name">{filename_display}</div>'
    
    if fingerprint:
        # Add core exploration info
        core_exploration = fingerprint.get('core_exploration', {})
        if core_exploration.get('central_question'):
            html += f'<div class="file-question">Q: {core_exploration["central_question"]}</div>'
        
        raw_essence = fingerprint.get('raw_essence')
        if raw_essence:
            html += f'<div class="file-essence">{raw_essence}</div>'
        
        if core_exploration.get('breakthrough_moment'):
            html += f'<div class="file-breakthrough">💡 {core_exploration["breakthrough_moment"]}</div>'
    else:
        html += '<div class="file-note">No semantic fingerprint available</div>'
    
    html += '</div>'
    return html


def generate_tree_html(tree: Dict[str, Any], cluster_metadata: Optional[Dict[str, Any]], semantic_fingerprints: Dict[str, Any]) -> str:
    """Generate the hierarchical tree HTML"""
    
    html = '<div class="tree-container">'
    
    # Sort coarse clusters by ID for consistent display
    for coarse_id in sorted(tree.keys()):
        coarse_data = tree[coarse_id]
        coarse_title, coarse_description = get_cluster_info(coarse_id, 'coarse', cluster_metadata)
        coarse_file_count = len(coarse_data['files'])
        
        html += f'''
        <div class="coarse-cluster">
            <div class="cluster-header coarse-header" onclick="toggleCluster('coarse_{coarse_id}')">
                <div class="cluster-info">
                    <h2 class="cluster-title">{coarse_title}</h2>
                    <div class="cluster-stats">
                        <span class="stat">{coarse_file_count} total items</span>
                        <span class="stat">{len(coarse_data["medium_clusters"])} sub-themes</span>
                    </div>
                </div>
                <div class="toggle-icon">▼</div>
            </div>
            
            {f'<div class="cluster-description">{coarse_description}</div>' if coarse_description else ''}
            
            <div class="cluster-content" id="coarse_{coarse_id}">
        '''
        
        # Process medium clusters within this coarse cluster
        for medium_id in sorted(coarse_data['medium_clusters'].keys()):
            medium_data = coarse_data['medium_clusters'][medium_id]
            medium_title, medium_description = get_cluster_info(medium_id, 'medium', cluster_metadata)
            medium_file_count = len(medium_data['files'])
            
            html += f'''
                <div class="medium-cluster">
                    <div class="cluster-header medium-header" onclick="toggleCluster('medium_{medium_id}')">
                        <div class="cluster-info">
                            <h3 class="cluster-title">{medium_title}</h3>
                            <div class="cluster-stats">
                                <span class="stat">{medium_file_count} items</span>
                                <span class="stat">{len(medium_data["fine_clusters"])} specific topics</span>
                            </div>
                        </div>
                        <div class="toggle-icon">▼</div>
                    </div>
                    
                    {f'<div class="cluster-description">{medium_description}</div>' if medium_description else ''}
                    
                    <div class="cluster-content" id="medium_{medium_id}">
            '''
            
            # Process fine clusters within this medium cluster
            for fine_id in sorted(medium_data['fine_clusters'].keys()):
                fine_data = medium_data['fine_clusters'][fine_id]
                fine_title, fine_description = get_cluster_info(fine_id, 'fine', cluster_metadata)
                fine_file_count = len(fine_data['files'])
                
                html += f'''
                    <div class="fine-cluster">
                        <div class="cluster-header fine-header" onclick="toggleCluster('fine_{fine_id}')">
                            <div class="cluster-info">
                                <h4 class="cluster-title">{fine_title}</h4>
                                <div class="cluster-stats">
                                    <span class="stat">{fine_file_count} voice memos</span>
                                </div>
                            </div>
                            <div class="toggle-icon">▼</div>
                        </div>
                        
                        {f'<div class="cluster-description">{fine_description}</div>' if fine_description else ''}
                        
                        <div class="cluster-content files-content" id="fine_{fine_id}">
                            <div class="files-grid">
                '''
                
                # Add the voice memo files
                for file_path in fine_data['files']:
                    html += generate_file_html(file_path, semantic_fingerprints)
                
                html += '''
                            </div>
                        </div>
                    </div>
                '''
            
            html += '''
                    </div>
                </div>
            '''
        
        html += '''
            </div>
        </div>
        '''
    
    html += '</div>'
    return html


def generate_html_content(tree: Dict[str, Any], intelligent_results: Dict[str, Any], semantic_fingerprints: Dict[str, Any], cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate the complete HTML content for the hierarchical tree website"""
    
    summary = intelligent_results.get('summary', {})
    
    tree_html = generate_tree_html(tree, cluster_metadata, semantic_fingerprints)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hierarchical Insight Tree Explorer</title>
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
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            font-size: 1.1rem;
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
            font-size: 1.8rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .summary-stat .label {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
        }}
        
        .tree-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .coarse-cluster {{
            margin-bottom: 30px;
            border: 2px solid #667eea;
            border-radius: 15px;
            overflow: hidden;
        }}
        
        .medium-cluster {{
            margin: 20px 0;
            margin-left: 30px;
            border: 2px solid #28a745;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .fine-cluster {{
            margin: 15px 0;
            margin-left: 30px;
            border: 2px solid #ffc107;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .coarse-header {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        
        .medium-header {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
        }}
        
        .fine-header {{
            background: linear-gradient(135deg, #ffc107, #fd7e14);
            color: white;
        }}
        
        .cluster-header:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}
        
        .cluster-info {{
            flex: 1;
        }}
        
        .cluster-title {{
            margin-bottom: 8px;
        }}
        
        .cluster-stats {{
            display: flex;
            gap: 15px;
        }}
        
        .stat {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .toggle-icon {{
            font-size: 1.5rem;
            transition: transform 0.3s ease;
        }}
        
        .cluster-header.collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}
        
        .cluster-description {{
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.05);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            font-style: italic;
            color: #555;
        }}
        
        .cluster-content {{
            padding: 20px;
            display: block;
            transition: all 0.3s ease;
        }}
        
        .cluster-content.hidden {{
            display: none;
        }}
        
        .files-content {{
            background: #f8f9fa;
        }}
        
        .files-grid {{
            display: grid;
            gap: 12px;
        }}
        
        .file-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .file-name {{
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 0.95rem;
        }}
        
        .file-question {{
            color: #667eea;
            font-style: italic;
            margin-bottom: 6px;
            font-size: 0.9rem;
        }}
        
        .file-essence {{
            color: #555;
            margin-bottom: 6px;
            font-size: 0.9rem;
        }}
        
        .file-breakthrough {{
            background: #e8f5e8;
            color: #2e7d32;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-top: 8px;
        }}
        
        .file-note {{
            color: #f57f17;
            font-size: 0.85rem;
            font-style: italic;
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
            
            .medium-cluster, .fine-cluster {{
                margin-left: 15px;
            }}
            
            .cluster-header {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Hierarchical Insight Tree</h1>
            <div class="subtitle">Navigate your voice memos through their natural semantic hierarchy</div>
            
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="number">{summary.get('total_analyzed', 0)}</div>
                    <div class="label">Voice Memos</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{len([c for c in tree.keys()])}</div>
                    <div class="label">Major Themes</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{sum(len(tree[c]['medium_clusters']) for c in tree.keys())}</div>
                    <div class="label">Sub-Themes</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{sum(len(tree[c]['medium_clusters'][m]['fine_clusters']) for c in tree.keys() for m in tree[c]['medium_clusters'].keys())}</div>
                    <div class="label">Specific Topics</div>
                </div>
            </div>
        </div>
        
        {tree_html}
    </div>

    <script>
        function toggleCluster(clusterId) {{
            const content = document.getElementById(clusterId);
            const header = content.previousElementSibling;
            
            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                header.classList.remove('collapsed');
            }} else {{
                content.classList.add('hidden');
                header.classList.add('collapsed');
            }}
        }}
        
        // Initialize all clusters as collapsed except the first coarse cluster
        document.addEventListener('DOMContentLoaded', function() {{
            const allContents = document.querySelectorAll('.cluster-content');
            const allHeaders = document.querySelectorAll('.cluster-header');
            
            // Collapse all clusters initially
            allContents.forEach((content, index) => {{
                if (index > 0) {{ // Keep first coarse cluster open
                    content.classList.add('hidden');
                }}
            }});
            
            allHeaders.forEach((header, index) => {{
                if (index > 0) {{
                    header.classList.add('collapsed');
                }}
            }});
        }});
    </script>
</body>
</html>
    """
    
    return html_content


def main():
    """Main function to generate the hierarchical tree website"""
    print("🌳 Generating Hierarchical Tree Website")
    print("=" * 50)
    
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
    
    # Build the hierarchical tree
    print("🌳 Building hierarchical tree structure...")
    tree = build_hierarchical_tree(intelligent_results)
    
    # Generate HTML content
    print("🔨 Generating HTML content...")
    html_content = generate_html_content(tree, intelligent_results, semantic_fingerprints, cluster_metadata)
    
    # Save the website
    output_file = Path("data/websites/hierarchical_tree_website.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Hierarchical tree website generated successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"🌐 Open in browser to explore your insight tree")
    
    # Show tree summary
    print(f"\n🌳 Tree Structure Summary:")
    coarse_count = len(tree)
    medium_count = sum(len(tree[c]['medium_clusters']) for c in tree.keys())
    fine_count = sum(len(tree[c]['medium_clusters'][m]['fine_clusters']) for c in tree.keys() for m in tree[c]['medium_clusters'].keys())
    
    print(f"  📊 {coarse_count} major themes")
    print(f"  📊 {medium_count} sub-themes") 
    print(f"  📊 {fine_count} specific topics")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())