#!/usr/bin/env python3
"""
Interactive Dendrogram Exploration Website Generator

Creates an interactive HTML website that allows users to cut the dendrogram at different heights
and explore what clusters are formed at each level.

Usage:
    python interactive_dendrogram_website.py

This will:
1. Load dendrogram cluster data from data/visualizations/dendrogram_clusters.json
2. Load semantic fingerprints from data/fingerprints/ directory 
3. Generate an interactive HTML website for exploring dendrogram cuts
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_dendrogram_clusters(file_path: str = "data/visualizations/dendrogram_clusters.json") -> Optional[Dict[str, Any]]:
    """Load dendrogram cluster data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("   Make sure you've run 'python mega_script.py visualize' first.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file {file_path}: {e}")
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


def generate_html_content(dendrogram_data: Dict[str, Any], semantic_fingerprints: Dict[str, Any]) -> str:
    """Generate the complete HTML content for the interactive dendrogram website"""
    
    metadata = dendrogram_data.get('metadata', {})
    levels = dendrogram_data.get('levels', {})
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Dendrogram Cluster Explorer</title>
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
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 0.9rem;
        }}
        
        .dendrogram-controls {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .controls-title {{
            font-size: 1.4rem;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .controls-title::before {{
            content: "🌳";
        }}
        
        .level-selector {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .level-button {{
            padding: 15px;
            background: #ecf0f1;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            text-align: left;
        }}
        
        .level-button:hover {{
            background: #d5dbdb;
        }}
        
        .level-button.active {{
            background: #3498db;
            color: white;
        }}
        
        .level-title {{
            font-weight: bold;
            display: block;
        }}
        
        .level-details {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }}
        
        .current-level-info {{
            background: #e8f6ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
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
        
        .cluster-display {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
        }}
        
        .cluster-display.active {{
            display: block;
        }}
        
        .cluster-content {{
            display: none;
        }}
        
        .cluster-content.active {{
            display: block;
        }}
        
        .cluster-header {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .cluster-title {{
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .cluster-summary {{
            color: #666;
            font-size: 0.95rem;
        }}
        
        .insights-section {{
            margin-bottom: 20px;
        }}
        
        .section-title {{
            font-size: 1.1rem;
            color: #2c3e50;
            margin-bottom: 10px;
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
        
        .insight-item {{
            background: #ffffff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #e74c3c;
        }}
        
        .insight-question {{
            font-size: 1rem;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .insight-essence {{
            font-size: 0.95rem;
            color: #34495e;
            margin-bottom: 8px;
            font-style: italic;
        }}
        
        .insight-source {{
            font-size: 0.85rem;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            padding-top: 8px;
        }}
        
        .files-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 8px;
        }}
        
        .file-item {{
            background: #ffffff;
            padding: 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            color: #2c3e50;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 40px;
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
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .level-selector {{
                grid-template-columns: 1fr;
            }}
            
            .cluster-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Interactive Dendrogram Explorer</h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{metadata.get('total_files', 0)}</div>
                    <div class="stat-label">Voice Memos</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(levels)}</div>
                    <div class="stat-label">Cut Levels</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metadata.get('linkage_method', 'ward').upper()}</div>
                    <div class="stat-label">Method</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metadata.get('distance_metric', 'cosine').upper()}</div>
                    <div class="stat-label">Distance</div>
                </div>
            </div>
        </div>
        
        <div class="dendrogram-controls">
            <h2 class="controls-title">Cut Dendrogram at Different Heights</h2>
            <div class="level-selector">
"""
    
    # Generate level selector buttons
    for level_name, level_data in levels.items():
        level_num = level_name.split('_')[1]
        html_content += f"""                <button class="level-button" onclick="showLevel('{level_name}')">
                    <span class="level-title">Level {level_num}</span>
                    <div class="level-details">{level_data['n_clusters']} clusters | Height: {level_data['cut_height']:.3f}</div>
                </button>
"""
    
    html_content += """            </div>
            <div id="current-level-info" class="current-level-info">
                Select a cut level above to explore clusters at different granularities
            </div>
        </div>
        
"""
    
    # Generate cluster content for each level
    for level_name, level_data in levels.items():
        clusters = level_data['clusters']
        html_content += f"""        <div id="{level_name}" class="cluster-display">
            <div class="cluster-tabs">
"""
        
        # Sort clusters by size
        cluster_items = [(int(cluster_id), files) for cluster_id, files in clusters.items()]
        cluster_items.sort(key=lambda x: (-len(x[1]), x[0]))
        
        # Generate cluster tabs
        for cluster_id, cluster_files in cluster_items:
            count = len(cluster_files)
            html_content += f"""                <button class="cluster-tab" onclick="showCluster('{level_name}', {cluster_id})">C{cluster_id} ({count})</button>
"""
        
        html_content += """            </div>
            
"""
        
        # Generate cluster content panels
        for cluster_id, cluster_files in cluster_items:
            insights = extract_cluster_insights(cluster_files, semantic_fingerprints)
            
            html_content += f"""            <div id="{level_name}-cluster-{cluster_id}" class="cluster-content">
                <div class="cluster-header">
                    <h3 class="cluster-title">🎯 Cluster {cluster_id}</h3>
                    <p class="cluster-summary">Collection of {len(cluster_files)} related voice memos</p>
                </div>
"""
            
            # Insights section
            if insights['unified_insights']:
                html_content += """                <div class="insights-section">
                    <h4 class="section-title">Key Insights</h4>
"""
                for insight_data in insights['unified_insights']:
                    html_content += """                    <div class="insight-item">
"""
                    if insight_data['question']:
                        html_content += f"""                        <div class="insight-question">❓ {insight_data['question']}</div>
"""
                    if insight_data['essence']:
                        html_content += f"""                        <div class="insight-essence">✨ {insight_data['essence']}</div>
"""
                    html_content += f"""                        <div class="insight-source">📄 Source: {insight_data['filename']}</div>
                    </div>
"""
                html_content += """                </div>
"""
            
            # Files section
            html_content += """                <div class="insights-section files-section">
                    <h4 class="section-title">Voice Memos</h4>
                    <div class="files-list">
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
    
    # Close HTML and add JavaScript
    html_content += """        <div class="footer">
            Interactive dendrogram cluster exploration tool
        </div>
    </div>
    
    <script>
        let currentLevel = null;
        
        function showLevel(levelName) {
            // Hide all level displays
            const displays = document.querySelectorAll('.cluster-display');
            displays.forEach(display => display.classList.remove('active'));
            
            // Remove active class from all level buttons
            const buttons = document.querySelectorAll('.level-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected level
            const targetDisplay = document.getElementById(levelName);
            if (targetDisplay) {
                targetDisplay.classList.add('active');
                currentLevel = levelName;
            }
            
            // Add active class to clicked button
            event.target.classList.add('active');
            
            // Update info display
            updateLevelInfo(levelName);
            
            // Show first cluster by default
            const firstClusterTab = targetDisplay.querySelector('.cluster-tab');
            if (firstClusterTab) {
                firstClusterTab.click();
            }
        }
        
        function showCluster(levelName, clusterId) {
            // Hide all cluster content for this level
            document.querySelectorAll(`#${levelName} .cluster-content`).forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all cluster tabs for this level
            document.querySelectorAll(`#${levelName} .cluster-tab`).forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected cluster content
            const targetContent = document.getElementById(`${levelName}-cluster-${clusterId}`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function updateLevelInfo(levelName) {
            const infoDisplay = document.getElementById('current-level-info');
            const levelData = getLevelData(levelName);
            
            if (levelData) {
                infoDisplay.innerHTML = `
                    <strong>${levelData.description}</strong><br>
                    Cut Height: ${levelData.cut_height.toFixed(3)} | 
                    ${levelData.n_clusters} clusters formed at this level<br>
                    <em>Click on a cluster tab above to explore its contents</em>
                `;
            }
        }
        
        function getLevelData(levelName) {
            const levels = """ + json.dumps({k: {
                'description': v['description'],
                'cut_height': v['cut_height'],
                'n_clusters': v['n_clusters']
            } for k, v in levels.items()}) + """;
            return levels[levelName];
        }
        
        // Show first level by default
        document.addEventListener('DOMContentLoaded', function() {
            const firstButton = document.querySelector('.level-button');
            if (firstButton) {
                firstButton.click();
            }
        });
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    """Main execution function"""
    print("🌳 Interactive Dendrogram Website Generator")
    print("=" * 60)
    
    # Load dendrogram cluster data
    print("📂 Loading dendrogram cluster data...")
    dendrogram_data = load_dendrogram_clusters()
    if dendrogram_data is None:
        return 1
    
    metadata = dendrogram_data.get('metadata', {})
    levels = dendrogram_data.get('levels', {})
    print(f"✅ Loaded dendrogram with {len(levels)} cut levels")
    print(f"   Total files: {metadata.get('total_files', 0)}")
    
    # Load semantic fingerprints
    print("\n📂 Loading semantic fingerprints...")
    semantic_fingerprints = load_semantic_fingerprints()
    
    # Generate HTML content
    print("🎨 Generating interactive HTML website...")
    html_content = generate_html_content(dendrogram_data, semantic_fingerprints)
    
    # Write to output file
    output_file = Path("data/websites/interactive_dendrogram.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Interactive dendrogram website generated!")
    print(f"📁 File saved to: {output_file.absolute()}")
    print(f"🌐 Open in browser: file://{output_file.absolute()}")
    print(f"🌳 Explore {len(levels)} different dendrogram cut levels")
    print(f"📊 Analyze clusters across {metadata.get('total_files', 0)} voice memos")
    
    return 0


if __name__ == "__main__":
    exit(main())