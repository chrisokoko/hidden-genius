#!/usr/bin/env python3
"""
Multi-Clustering Website Generator

Creates an interactive HTML website that allows you to compare different clustering results
side-by-side to find the most semantically meaningful groupings.

Usage:
    python multi_clustering_website_generator.py

This will:
1. Load all clustering attempts from data/clusters/ directory
2. Load semantic fingerprints from data/fingerprints/ directory 
3. Generate an interactive HTML website with tabs to switch between clustering approaches
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_all_clustering_attempts(clusters_dir: str = "data/clusters") -> List[Dict[str, Any]]:
    """Load all clustering attempts from the clusters directory"""
    clusters_path = Path(clusters_dir)
    
    if not clusters_path.exists():
        print(f"❌ Clusters directory not found: {clusters_dir}")
        print("   Run 'python mega_script.py cluster' first.")
        return []
    
    # Load summary first to get ordering
    summary_path = clusters_path / "all_attempts_summary.json"
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        return []
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    attempts = []
    print(f"📂 Loading {summary['total_attempts']} clustering attempts...")
    
    for attempt_info in summary['attempts']:
        attempt_path = clusters_path / attempt_info['filename']
        if attempt_path.exists():
            try:
                with open(attempt_path, 'r', encoding='utf-8') as f:
                    attempt_data = json.load(f)
                # Add metadata from summary
                attempt_data['metadata'] = attempt_info
                attempts.append(attempt_data)
            except Exception as e:
                print(f"⚠️  Error loading {attempt_path}: {e}")
    
    print(f"✅ Loaded {len(attempts)} clustering attempts")
    return attempts


def load_semantic_fingerprints(fingerprints_dir: str = "data/fingerprints") -> Dict[str, Any]:
    """Load all semantic fingerprints from the fingerprints directory"""
    fingerprints = {}
    fingerprints_path = Path(fingerprints_dir)
    
    if not fingerprints_path.exists():
        print(f"⚠️  Fingerprints directory not found: {fingerprints_dir}")
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
        'missing_files': []
    }
    
    for filename in cluster_filenames:
        base_filename = filename.replace('.json', '')
        fingerprint = None
        
        possible_keys = [filename, base_filename, f"{base_filename}.json"]
        for key in possible_keys:
            if key in semantic_fingerprints:
                fingerprint = semantic_fingerprints[key]
                break
        
        if fingerprint:
            central_question = None
            if 'core_exploration' in fingerprint and 'central_question' in fingerprint['core_exploration']:
                central_question = fingerprint['core_exploration']['central_question']
            
            raw_essence = None
            if 'raw_essence' in fingerprint:
                raw_essence = fingerprint['raw_essence']
            elif 'core_exploration' in fingerprint and 'raw_essence' in fingerprint['core_exploration']:
                raw_essence = fingerprint['core_exploration']['raw_essence']
            
            if central_question or raw_essence:
                insights['unified_insights'].append({
                    'question': central_question,
                    'essence': raw_essence,
                    'filename': filename,
                    'full_semantic_fingerprint': fingerprint
                })
        else:
            insights['missing_files'].append(filename)
    
    return insights


def generate_html_content(clustering_attempts: List[Dict[str, Any]], semantic_fingerprints: Dict[str, Any]) -> str:
    """Generate the complete HTML content for the multi-clustering website"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Clustering Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }}
        
        .clustering-selector {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .selector-title {{
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .selector-title::before {{
            content: "🔄";
        }}
        
        .clustering-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .clustering-tab {{
            padding: 12px 20px;
            background: #ecf0f1;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            min-width: 200px;
            text-align: left;
        }}
        
        .clustering-tab:hover {{
            background: #d5dbdb;
        }}
        
        .clustering-tab.active {{
            background: #3498db;
            color: white;
        }}
        
        .tab-method {{
            font-weight: bold;
            display: block;
        }}
        
        .tab-details {{
            font-size: 0.8rem;
            opacity: 0.8;
            margin-top: 4px;
        }}
        
        .clustering-content {{
            display: none;
        }}
        
        .clustering-content.active {{
            display: block;
        }}
        
        .clustering-stats {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.3rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
        }}
        
        .cluster-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 20px;
            background: white;
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .cluster-tab {{
            padding: 8px 16px;
            background: #f8f9fa;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            flex: 1;
            min-width: 100px;
        }}
        
        .cluster-tab:hover {{
            background: #e9ecef;
        }}
        
        .cluster-tab.active {{
            background: #28a745;
            color: white;
        }}
        
        .noise-tab {{
            background: #e74c3c !important;
            color: white !important;
        }}
        
        .noise-tab.active {{
            background: #c0392b !important;
        }}
        
        .cluster-content {{
            display: none;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .cluster-content.active {{
            display: block;
        }}
        
        .cluster-header {{
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .cluster-title {{
            font-size: 1.6rem;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        
        .cluster-summary {{
            color: #666;
            font-size: 1rem;
        }}
        
        .insights-section {{
            margin-bottom: 30px;
        }}
        
        .section-title {{
            font-size: 1.2rem;
            color: #2c3e50;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .insights-section .section-title::before {{
            content: "💡";
        }}
        
        .files-section .section-title::before {{
            content: "📄";
        }}
        
        .unified-insight-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #9b59b6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .insight-question {{
            font-size: 1.1rem;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        
        .insight-essence {{
            font-size: 1rem;
            color: #34495e;
            margin-bottom: 8px;
            line-height: 1.4;
            font-style: italic;
        }}
        
        .insight-source {{
            font-size: 0.85rem;
            color: #7f8c8d;
            padding-top: 6px;
            border-top: 1px solid #ecf0f1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .dropdown-toggle {{
            background: #3498db;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: background-color 0.3s ease;
        }}
        
        .dropdown-toggle:hover {{
            background: #2980b9;
        }}
        
        .semantic-dropdown {{
            display: none;
            background: #ffffff;
            border: 1px solid #e1e8ed;
            border-radius: 6px;
            margin-top: 10px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .semantic-dropdown.show {{
            display: block;
        }}
        
        .semantic-section {{
            padding: 12px 15px;
            border-bottom: 1px solid #f1f3f4;
        }}
        
        .semantic-section:last-child {{
            border-bottom: none;
        }}
        
        .semantic-title {{
            font-weight: 600;
            color: #2c3e50;
            font-size: 0.9rem;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .semantic-content {{
            color: #34495e;
            font-size: 0.85rem;
            line-height: 1.4;
            white-space: pre-wrap;
        }}
        
        .semantic-array {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 5px;
        }}
        
        .semantic-tag {{
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75rem;
            color: #7f8c8d;
        }}
        
        .files-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 8px;
        }}
        
        .file-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            color: #2c3e50;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 30px;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .clustering-tabs, .cluster-tabs {{
                flex-direction: column;
            }}
            
            .clustering-tab, .cluster-tab {{
                flex: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Multi-Clustering Comparison</h1>
            <p>Compare different clustering approaches to find the most semantically meaningful groupings</p>
        </div>
        
        <div class="clustering-selector">
            <h2 class="selector-title">Select Clustering Approach</h2>
            <div class="clustering-tabs">
"""
    
    # Generate clustering approach tabs
    for i, attempt in enumerate(clustering_attempts):
        metadata = attempt['metadata']
        method = metadata['method']
        n_clusters = metadata['n_clusters']
        quality_score = metadata['quality_score']
        
        # Create readable params string
        params = metadata['params']
        params_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in params.items()])
        
        html_content += f"""                <button class="clustering-tab" onclick="showClustering({i})">
                    <div class="tab-method">{method.upper()} - {n_clusters} clusters</div>
                    <div class="tab-details">Score: {quality_score:.3f} | {params_str}</div>
                </button>
"""
    
    html_content += """            </div>
        </div>
        
"""
    
    # Generate content for each clustering approach
    for i, attempt in enumerate(clustering_attempts):
        clusters = attempt['clusters']
        metadata = attempt['metadata']
        
        html_content += f"""        <div id="clustering-{i}" class="clustering-content">
            <div class="clustering-stats">
                <div class="stat-item">
                    <div class="stat-value">{metadata['n_clusters']}</div>
                    <div class="stat-label">Clusters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metadata['quality_score']:.3f}</div>
                    <div class="stat-label">Quality Score</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metadata['method'].upper()}</div>
                    <div class="stat-label">Method</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len([c for c in clusters.keys() if int(c) != -1])}</div>
                    <div class="stat-label">Real Clusters</div>
                </div>
            </div>
            
            <div class="cluster-tabs">
"""
        
        # Sort clusters by size
        cluster_items = [(int(cluster_id), files) for cluster_id, files in clusters.items()]
        cluster_items.sort(key=lambda x: (-len(x[1]), x[0]))  # Sort by size DESC, then ID ASC
        
        for cluster_id, cluster_files in cluster_items:
            count = len(cluster_files)
            if cluster_id == -1:
                html_content += f"""                <button class="cluster-tab noise-tab" onclick="showCluster({i}, {cluster_id})">Noise ({count})</button>
"""
            else:
                html_content += f"""                <button class="cluster-tab" onclick="showCluster({i}, {cluster_id})">C{cluster_id} ({count})</button>
"""
        
        html_content += """            </div>
            
"""
        
        # Generate cluster content panels for this clustering approach
        for cluster_id, cluster_files in cluster_items:
            insights = extract_cluster_insights(cluster_files, semantic_fingerprints)
            
            html_content += f"""            <div id="clustering-{i}-cluster-{cluster_id}" class="cluster-content">
"""
            
            if cluster_id == -1:
                html_content += f"""                <div class="cluster-header">
                    <h3 class="cluster-title">🌫️ Noise Points</h3>
                    <p class="cluster-summary">Items that don't fit clearly into any cluster ({len(cluster_files)} items)</p>
                </div>
"""
            else:
                html_content += f"""                <div class="cluster-header">
                    <h3 class="cluster-title">🎯 Cluster {cluster_id}</h3>
                    <p class="cluster-summary">Collection of {len(cluster_files)} related voice memos</p>
                </div>
"""
            
            # Insights section
            if insights['unified_insights']:
                html_content += """                <div class="insights-section">
                    <h4 class="section-title">Key Insights</h4>
"""
                for j, insight_data in enumerate(insights['unified_insights']):
                    insight_id = f"clustering-{i}-cluster-{cluster_id}-insight-{j}"
                    html_content += f"""                    <div class="unified-insight-card">
"""
                    if insight_data['essence']:
                        html_content += f"""                        <div class="insight-essence">✨ {insight_data['essence']}</div>
"""
                    html_content += f"""                        <div class="insight-source">
                            <span>📄 {insight_data['filename']}</span>
                            <button class="dropdown-toggle" onclick="toggleSemanticDropdown('{insight_id}')">
                                View Details
                            </button>
                        </div>
                        <div id="{insight_id}" class="semantic-dropdown">
"""
                    
                    # Add semantic fingerprint sections
                    fingerprint = insight_data.get('full_semantic_fingerprint', {})
                    
                    # Core exploration section - include central question here
                    if 'core_exploration' in fingerprint:
                        core = fingerprint['core_exploration']
                        html_content += """                            <div class="semantic-section">
                                <div class="semantic-title">Core Exploration</div>
"""
                        if 'central_question' in core:
                            html_content += f"""                                <div class="semantic-content"><strong>Central Question:</strong> {core['central_question']}</div>
"""
                        if 'key_tension' in core:
                            html_content += f"""                                <div class="semantic-content"><strong>Key Tension:</strong> {core['key_tension']}</div>
"""
                        if 'breakthrough_moment' in core:
                            html_content += f"""                                <div class="semantic-content"><strong>Breakthrough Moment:</strong> {core['breakthrough_moment']}</div>
"""
                        if 'edge_of_understanding' in core:
                            html_content += f"""                                <div class="semantic-content"><strong>Edge of Understanding:</strong> {core['edge_of_understanding']}</div>
"""
                        html_content += """                            </div>
"""
                    
                    # Raw essence (separate prominent section)
                    if 'raw_essence' in fingerprint:
                        html_content += f"""                            <div class="semantic-section">
                                <div class="semantic-title">Raw Essence</div>
                                <div class="semantic-content">{fingerprint['raw_essence']}</div>
                            </div>
"""
                    
                    # Conceptual DNA
                    if 'conceptual_dna' in fingerprint:
                        html_content += """                            <div class="semantic-section">
                                <div class="semantic-title">Conceptual DNA</div>
"""
                        for k, concept in enumerate(fingerprint['conceptual_dna'], 1):
                            html_content += f"""                                <div class="semantic-content">{k}. {concept}</div>
"""
                        html_content += """                            </div>
"""
                    
                    # Insight pattern
                    if 'insight_pattern' in fingerprint:
                        pattern = fingerprint['insight_pattern']
                        html_content += """                            <div class="semantic-section">
                                <div class="semantic-title">Insight Pattern</div>
"""
                        if 'thinking_styles' in pattern:
                            html_content += """                                <div class="semantic-content"><strong>Thinking Styles:</strong></div>
                                <div class="semantic-array">
"""
                            for style in pattern['thinking_styles']:
                                html_content += f"""                                    <span class="semantic-tag">{style}</span>
"""
                            html_content += """                                </div>
"""
                        if 'insight_type' in pattern:
                            html_content += f"""                                <div class="semantic-content"><strong>Insight Type:</strong> {pattern['insight_type']}</div>
"""
                        if 'development_stage' in pattern:
                            html_content += f"""                                <div class="semantic-content"><strong>Development Stage:</strong> {pattern['development_stage']}</div>
"""
                        if 'connected_domains' in pattern:
                            html_content += """                                <div class="semantic-content"><strong>Connected Domains:</strong></div>
                                <div class="semantic-array">
"""
                            for domain in pattern['connected_domains']:
                                html_content += f"""                                    <span class="semantic-tag">{domain}</span>
"""
                            html_content += """                                </div>
"""
                        html_content += """                            </div>
"""
                    
                    # Insight quality
                    if 'insight_quality' in fingerprint:
                        quality = fingerprint['insight_quality']
                        html_content += """                            <div class="semantic-section">
                                <div class="semantic-title">Insight Quality</div>
"""
                        if 'uniqueness_score' in quality:
                            html_content += f"""                                <div class="semantic-content"><strong>Uniqueness Score:</strong> {quality['uniqueness_score']:.2f}</div>
"""
                        if 'depth_score' in quality:
                            html_content += f"""                                <div class="semantic-content"><strong>Depth Score:</strong> {quality['depth_score']:.2f}</div>
"""
                        if 'generative_score' in quality:
                            html_content += f"""                                <div class="semantic-content"><strong>Generative Score:</strong> {quality['generative_score']:.2f}</div>
"""
                        if 'usefulness_score' in quality:
                            html_content += f"""                                <div class="semantic-content"><strong>Usefulness Score:</strong> {quality['usefulness_score']:.2f}</div>
"""
                        if 'confidence_score' in quality:
                            html_content += f"""                                <div class="semantic-content"><strong>Confidence Score:</strong> {quality['confidence_score']:.2f}</div>
"""
                        html_content += """                            </div>
"""
                    
                    html_content += """                        </div>
                    </div>
"""
                html_content += """                </div>
"""
            
            # Files section
            html_content += """                <div class="insights-section files-section">
                    <h4 class="section-title">Voice Memos</h4>
                    <div class="files-grid">
"""
            for filename in cluster_files:
                html_content += f"""                        <div class="file-item">{filename}</div>
"""
            html_content += """                    </div>
                </div>
"""
            
            if insights['missing_files']:
                html_content += f"""                <div class="empty-state">
                    ⚠️ {len(insights['missing_files'])} files missing semantic fingerprints
                </div>
"""
            
            html_content += """            </div>
"""
        
        html_content += """        </div>
"""
    
    # JavaScript for interaction
    html_content += """        <div class="footer">
            Multi-clustering comparison tool
        </div>
    </div>
    
    <script>
        let currentClustering = 0;
        
        function showClustering(clusteringId) {
            // Hide all clustering content
            document.querySelectorAll('.clustering-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all clustering tabs
            document.querySelectorAll('.clustering-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected clustering
            const targetContent = document.getElementById(`clustering-${clusteringId}`);
            if (targetContent) {
                targetContent.classList.add('active');
                currentClustering = clusteringId;
            }
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Show first cluster by default
            const firstClusterTab = targetContent.querySelector('.cluster-tab');
            if (firstClusterTab) {
                firstClusterTab.click();
            }
        }
        
        function showCluster(clusteringId, clusterId) {
            // Hide all cluster content for this clustering
            document.querySelectorAll(`#clustering-${clusteringId} .cluster-content`).forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all cluster tabs for this clustering
            document.querySelectorAll(`#clustering-${clusteringId} .cluster-tab`).forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected cluster content
            const targetContent = document.getElementById(`clustering-${clusteringId}-cluster-${clusterId}`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function toggleSemanticDropdown(dropdownId) {
            const dropdown = document.getElementById(dropdownId);
            const button = event.target;
            
            if (dropdown.classList.contains('show')) {
                dropdown.classList.remove('show');
                button.textContent = 'View Details';
            } else {
                // Close all other dropdowns first
                document.querySelectorAll('.semantic-dropdown.show').forEach(dd => {
                    dd.classList.remove('show');
                    const otherButton = dd.previousElementSibling.querySelector('.dropdown-toggle');
                    if (otherButton) otherButton.textContent = 'View Details';
                });
                
                dropdown.classList.add('show');
                button.textContent = 'Hide Details';
            }
        }
        
        // Initialize - show first clustering by default
        document.addEventListener('DOMContentLoaded', function() {
            const firstClusteringTab = document.querySelector('.clustering-tab');
            if (firstClusteringTab) {
                firstClusteringTab.click();
            }
        });
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    """Main execution function"""
    print("🔍 Multi-Clustering Website Generator")
    print("=" * 60)
    
    # Load all clustering attempts
    clustering_attempts = load_all_clustering_attempts()
    if not clustering_attempts:
        print("❌ No clustering attempts found. Run 'python mega_script.py cluster' first.")
        return 1
    
    print(f"✅ Loaded {len(clustering_attempts)} different clustering approaches")
    
    # Load semantic fingerprints
    print("\n📂 Loading semantic fingerprints...")
    semantic_fingerprints = load_semantic_fingerprints()
    
    # Generate HTML content
    print("🎨 Generating multi-clustering comparison website...")
    html_content = generate_html_content(clustering_attempts, semantic_fingerprints)
    
    # Write to output file
    output_file = Path("data/visualizations/multi_clustering_website.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Multi-clustering website generated successfully!")
    print(f"📁 File saved to: {output_file.absolute()}")
    print(f"🌐 Open in browser: file://{output_file.absolute()}")
    print(f"🔄 Compare {len(clustering_attempts)} different clustering approaches")
    print(f"💡 Find the most semantically meaningful groupings!")
    
    return 0


if __name__ == "__main__":
    exit(main())