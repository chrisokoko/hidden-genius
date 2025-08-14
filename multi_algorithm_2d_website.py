#!/usr/bin/env python3
"""
Multi-Algorithm 2D Visualization Comparison Website

Creates an interactive HTML website that allows users to compare different clustering algorithms
on t-SNE and UMAP plots with cluster labels and interactive exploration.

Usage:
    python multi_algorithm_2d_website.py

This will:
1. Load coordinate data from data/visualizations/ directory
2. Load semantic fingerprints from data/fingerprints/ directory 
3. Generate an interactive HTML website for comparing clustering approaches
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_multi_algorithm_summary(file_path: str = "data/visualizations/multi_algorithm_summary.json") -> Optional[Dict[str, Any]]:
    """Load multi-algorithm summary data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("   Make sure you've run 'python mega_script.py multi-viz' first.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file {file_path}: {e}")
        return None


def load_coordinate_data(viz_dir: Path, coordinate_files: List[str]) -> Dict[str, Any]:
    """Load all coordinate data files"""
    coord_data = {}
    
    for coord_file in coordinate_files:
        file_path = viz_dir / coord_file
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    coord_data[coord_file] = data
            except Exception as e:
                print(f"⚠️  Error loading {coord_file}: {e}")
    
    print(f"✅ Loaded {len(coord_data)} coordinate datasets")
    return coord_data


def load_semantic_fingerprints(fingerprints_dir: str = "data/fingerprints") -> Dict[str, Any]:
    """Load all semantic fingerprints"""
    fingerprints = {}
    fingerprints_path = Path(fingerprints_dir)
    
    if not fingerprints_path.exists():
        print(f"⚠️  Fingerprints directory not found: {fingerprints_dir}")
        return fingerprints
    
    json_files = list(fingerprints_path.rglob("*.json"))
    
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


def generate_html_content(summary_data: Dict[str, Any], coord_data: Dict[str, Any], semantic_fingerprints: Dict[str, Any]) -> str:
    """Generate the complete HTML content for the multi-algorithm comparison website"""
    
    meaningful_clusterings = summary_data.get('meaningful_clusterings', [])
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Algorithm 2D Clustering Comparison</title>
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
            max-width: 1600px;
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
        
        .controls-panel {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .controls-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            align-items: start;
        }}
        
        .algorithm-selector {{
            display: flex;
            flex-direction: column;
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
        
        .algorithm-buttons {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .algorithm-button {{
            padding: 15px;
            background: #ecf0f1;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            text-align: left;
        }}
        
        .algorithm-button:hover {{
            background: #d5dbdb;
        }}
        
        .algorithm-button.active {{
            background: #3498db;
            color: white;
        }}
        
        .algorithm-title {{
            font-weight: bold;
            display: block;
        }}
        
        .algorithm-details {{
            font-size: 0.8rem;
            opacity: 0.8;
            margin-top: 5px;
        }}
        
        .view-selector {{
            display: flex;
            flex-direction: column;
        }}
        
        .view-title {{
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .view-title::before {{
            content: "🗺️";
        }}
        
        .view-buttons {{
            display: flex;
            gap: 10px;
        }}
        
        .view-button {{
            padding: 15px 25px;
            background: #ecf0f1;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            flex: 1;
        }}
        
        .view-button:hover {{
            background: #d5dbdb;
        }}
        
        .view-button.active {{
            background: #e74c3c;
            color: white;
        }}
        
        .current-selection {{
            background: #e8f6ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin-top: 20px;
        }}
        
        .visualization-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .viz-content {{
            display: none;
            min-height: 600px;
        }}
        
        .viz-content.active {{
            display: block;
        }}
        
        .plot-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            background: #fafafa;
        }}
        
        .plot-svg {{
            width: 100%;
            height: 100%;
        }}
        
        .cluster-point {{
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .cluster-point:hover {{
            r: 6;
            stroke-width: 2;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            display: none;
        }}
        
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        
        .legend-items {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            font-size: 1.2rem;
            color: #7f8c8d;
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 1200px) {{
            .controls-grid {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .algorithm-buttons {{
                grid-template-columns: 1fr;
            }}
            
            .view-buttons {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Multi-Algorithm 2D Clustering Comparison</h1>
            <p>Compare different clustering approaches on t-SNE and UMAP visualizations</p>
        </div>
        
        <div class="controls-panel">
            <div class="controls-grid">
                <div class="algorithm-selector">
                    <h2 class="selector-title">Select Clustering Algorithm</h2>
                    <div class="algorithm-buttons">
"""
    
    # Generate algorithm selector buttons
    for i, clustering in enumerate(meaningful_clusterings):
        method = clustering['method']
        n_clusters = clustering['n_clusters']
        quality_score = clustering['quality_score']
        
        html_content += f"""                        <button class="algorithm-button" onclick="selectAlgorithm('{method}_n{n_clusters}')">
                            <span class="algorithm-title">{method.upper()} - {n_clusters} clusters</span>
                            <div class="algorithm-details">Quality: {quality_score:.3f}</div>
                        </button>
"""
    
    html_content += """                    </div>
                </div>
                
                <div class="view-selector">
                    <h2 class="view-title">Select Visualization Method</h2>
                    <div class="view-buttons">
                        <button class="view-button" onclick="selectView('tsne')">t-SNE</button>
                        <button class="view-button" onclick="selectView('umap')">UMAP</button>
                    </div>
                    
                    <div id="current-selection" class="current-selection">
                        Select an algorithm and visualization method to begin
                    </div>
                </div>
            </div>
        </div>
        
        <div class="visualization-container">
            <div id="viz-loading" class="loading">
                Select an algorithm and view type to display visualization
            </div>
            
            <div id="viz-content" class="viz-content">
                <div class="plot-container">
                    <svg class="plot-svg" id="main-plot">
                    </svg>
                    <div class="tooltip" id="tooltip"></div>
                </div>
                <div class="legend" id="legend" style="display: none;">
                    <div class="legend-title">Cluster Legend</div>
                    <div class="legend-items" id="legend-items"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Multi-algorithm clustering comparison tool
        </div>
    </div>
    
    <script>
        // Store all coordinate data
        const coordinateData = """ + json.dumps(coord_data) + """;
        
        // Store semantic fingerprints
        const semanticFingerprints = """ + json.dumps(semantic_fingerprints) + """;
        
        let currentAlgorithm = null;
        let currentView = null;
        
        function selectAlgorithm(algorithmKey) {
            currentAlgorithm = algorithmKey;
            
            // Update button states
            document.querySelectorAll('.algorithm-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            updateVisualization();
            updateSelectionDisplay();
        }
        
        function selectView(viewType) {
            currentView = viewType;
            
            // Update button states
            document.querySelectorAll('.view-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            updateVisualization();
            updateSelectionDisplay();
        }
        
        function updateSelectionDisplay() {
            const selectionDiv = document.getElementById('current-selection');
            if (currentAlgorithm && currentView) {
                selectionDiv.innerHTML = `
                    <strong>Current Selection:</strong><br>
                    Algorithm: ${currentAlgorithm.replace('_', ' ').toUpperCase()}<br>
                    Visualization: ${currentView.toUpperCase()}
                `;
            } else {
                selectionDiv.innerHTML = 'Select an algorithm and visualization method to begin';
            }
        }
        
        function updateVisualization() {
            if (!currentAlgorithm || !currentView) {
                document.getElementById('viz-loading').style.display = 'flex';
                document.getElementById('viz-content').classList.remove('active');
                return;
            }
            
            const dataKey = `${currentAlgorithm}_${currentView}_coordinates.json`;
            const data = coordinateData[dataKey];
            
            if (!data) {
                document.getElementById('viz-loading').innerHTML = 'Data not available for this combination';
                document.getElementById('viz-loading').style.display = 'flex';
                document.getElementById('viz-content').classList.remove('active');
                return;
            }
            
            document.getElementById('viz-loading').style.display = 'none';
            document.getElementById('viz-content').classList.add('active');
            
            renderPlot(data);
        }
        
        function renderPlot(data) {
            const svg = document.getElementById('main-plot');
            const tooltip = document.getElementById('tooltip');
            const legend = document.getElementById('legend');
            const legendItems = document.getElementById('legend-items');
            
            // Clear previous content
            svg.innerHTML = '';
            legendItems.innerHTML = '';
            
            const coordinates = data.coordinates;
            const labels = data.labels;
            const filenames = data.filenames;
            
            // Calculate plot dimensions
            const margin = 50;
            const width = 800;
            const height = 600;
            const plotWidth = width - 2 * margin;
            const plotHeight = height - 2 * margin;
            
            // Find coordinate ranges
            const xCoords = coordinates.map(c => c[0]);
            const yCoords = coordinates.map(c => c[1]);
            const xMin = Math.min(...xCoords);
            const xMax = Math.max(...xCoords);
            const yMin = Math.min(...yCoords);
            const yMax = Math.max(...yCoords);
            
            // Create scales
            const xScale = d3.scaleLinear()
                .domain([xMin, xMax])
                .range([margin, width - margin]);
            const yScale = d3.scaleLinear()
                .domain([yMin, yMax])
                .range([height - margin, margin]);
            
            // Create color scale
            const uniqueLabels = [...new Set(labels)];
            const colors = d3.scaleOrdinal(d3.schemeCategory10);
            
            // Create SVG group
            const g = d3.select(svg).append('g');
            
            // Add points
            coordinates.forEach((coord, i) => {
                const x = xScale(coord[0]);
                const y = yScale(coord[1]);
                const label = labels[i];
                const filename = filenames[i];
                const color = label === -1 ? '#000000' : colors(label);
                
                const circle = g.append('circle')
                    .attr('cx', x)
                    .attr('cy', y)
                    .attr('r', 4)
                    .attr('fill', color)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1)
                    .attr('class', 'cluster-point')
                    .attr('data-label', label)
                    .attr('data-filename', filename);
                
                // Add hover events
                circle.on('mouseover', function(event) {
                    const rect = svg.getBoundingClientRect();
                    const tooltipText = `Cluster: ${label === -1 ? 'Noise' : label}<br>File: ${filename}`;
                    
                    tooltip.innerHTML = tooltipText;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX - rect.left + 10) + 'px';
                    tooltip.style.top = (event.clientY - rect.top - 10) + 'px';
                })
                .on('mouseout', function() {
                    tooltip.style.display = 'none';
                });
            });
            
            // Add axes
            const xAxis = d3.axisBottom(xScale);
            const yAxis = d3.axisLeft(yScale);
            
            g.append('g')
                .attr('transform', `translate(0, ${height - margin})`)
                .call(xAxis);
            
            g.append('g')
                .attr('transform', `translate(${margin}, 0)`)
                .call(yAxis);
            
            // Add axis labels
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height - 10)
                .attr('text-anchor', 'middle')
                .text(`${currentView.toUpperCase()} Component 1`);
            
            g.append('text')
                .attr('transform', 'rotate(-90)')
                .attr('x', -height / 2)
                .attr('y', 15)
                .attr('text-anchor', 'middle')
                .text(`${currentView.toUpperCase()} Component 2`);
            
            // Update legend
            uniqueLabels.forEach(label => {
                const color = label === -1 ? '#000000' : colors(label);
                const count = labels.filter(l => l === label).length;
                const labelText = label === -1 ? 'Noise' : `Cluster ${label}`;
                
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background-color: ${color}"></div>
                    <span>${labelText} (${count})</span>
                `;
                legendItems.appendChild(item);
            });
            
            legend.style.display = 'block';
        }
        
        // Load D3.js
        const d3Script = document.createElement('script');
        d3Script.src = 'https://d3js.org/d3.v7.min.js';
        document.head.appendChild(d3Script);
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Auto-select first algorithm and t-SNE
            setTimeout(() => {
                const firstButton = document.querySelector('.algorithm-button');
                const tsneButton = document.querySelector('.view-button[onclick*="tsne"]');
                
                if (firstButton && tsneButton) {
                    firstButton.click();
                    tsneButton.click();
                }
            }, 100);
        });
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    """Main execution function"""
    print("🎯 Multi-Algorithm 2D Visualization Website Generator")
    print("=" * 60)
    
    # Load summary data
    print("📂 Loading multi-algorithm summary...")
    summary_data = load_multi_algorithm_summary()
    if summary_data is None:
        return 1
    
    meaningful_clusterings = summary_data.get('meaningful_clusterings', [])
    coordinate_files = summary_data.get('coordinate_files', [])
    print(f"✅ Found {len(meaningful_clusterings)} clustering algorithms")
    print(f"   Coordinate files: {len(coordinate_files)}")
    
    # Load coordinate data
    print("\n📂 Loading coordinate data...")
    viz_dir = Path("data/visualizations")
    coord_data = load_coordinate_data(viz_dir, coordinate_files)
    
    # Load semantic fingerprints
    print("\n📂 Loading semantic fingerprints...")
    semantic_fingerprints = load_semantic_fingerprints()
    
    # Generate HTML content
    print("🎨 Generating interactive comparison website...")
    html_content = generate_html_content(summary_data, coord_data, semantic_fingerprints)
    
    # Write to output file
    output_file = Path("data/websites/multi_algorithm_2d_comparison.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Multi-algorithm comparison website generated!")
    print(f"📁 File saved to: {output_file.absolute()}")
    print(f"🌐 Open in browser: file://{output_file.absolute()}")
    print(f"🔄 Compare {len(meaningful_clusterings)} clustering algorithms")
    print(f"🎯 Interactive t-SNE and UMAP visualizations")
    print(f"💡 Hover over points to see insights!")
    
    return 0


if __name__ == "__main__":
    exit(main())