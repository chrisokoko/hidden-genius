# Hidden Genius: Voice Memo Processing Pipeline

A complete voice memo processing and clustering analysis system with interactive visualizations.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements_megascript.txt
```

### Usage
```bash
# Full pipeline: transcribe → fingerprint → embeddings → intelligent clustering
python mega_script.py all

# Individual steps
python mega_script.py transcribe     # Transcribe audio files
python mega_script.py fingerprint    # Generate semantic fingerprints  
python mega_script.py embeddings     # Create OpenAI embeddings
python mega_script.py cluster        # Run intelligent hierarchical clustering
python mega_script.py visualize      # Generate basic visualizations

# Generate interactive websites
python intelligent_hierarchical_website.py        # NEW: Intelligent semantic hierarchy explorer
python interactive_dendrogram_website.py           # Dendrogram explorer
python multi_algorithm_2d_website.py              # 2D algorithm comparison
python clustering_website_generator.py            # Single clustering website
python multi_clustering_website_generator.py      # Multi-clustering comparison
```

## 📁 Project Structure

```
hidden_genius/
├── mega_script.py                              # Main processing pipeline
├── *website*.py                               # Website generators
├── requirements_megascript.txt                # Dependencies
├── .env                                       # API keys
├── audio_files/                              # Voice memo audio files
│   ├── 00_under_10_seconds/
│   ├── 01_10s_to_1min/
│   ├── 02_1min_to_5min/
│   └── 03_5min_to_20min/
└── data/                                     # Generated data
    ├── transcripts/                          # Transcribed text
    ├── fingerprints/                         # Semantic fingerprints
    ├── embeddings/                           # OpenAI embeddings
    ├── clusters/                             # Clustering results
    ├── visualizations/                       # PNG plots & coordinate data
    └── websites/                            # Interactive HTML sites
```

## 🎯 Interactive Visualizations

### 1. **🧠 Intelligent Hierarchical Explorer** (`data/websites/intelligent_hierarchical_website.html`) **[NEW!]**
- **Three-Phase Intelligent Clustering**: Natural breakpoints → Quality validation → Level assignment
- **Semantic Hierarchy Levels**:
  - **Major Themes** (3-8 clusters): High-level conceptual groupings
  - **Sub-Themes** (8-20 clusters): Balanced granularity and coherence  
  - **Specific Topics** (20-60 clusters): Detailed topical clusters
- **Quality Metrics**: Silhouette, balance, separation, and stability scores
- **Interactive Exploration**: Click into each level to explore clusters with semantic insights
- **Natural Breakpoint Detection**: Uses elbow method, curvature analysis, and change point detection

### 2. **Dendrogram Explorer** (`data/websites/interactive_dendrogram.html`)
- Cut dendrogram at 6 different heights (18-247 clusters)
- Click through cluster tabs to explore insights
- View semantic fingerprints and voice memo contents

### 3. **Multi-Algorithm Comparison** (`data/websites/multi_algorithm_2d_comparison.html`)
- Compare 8 clustering algorithms (67-296 clusters)
- Toggle between t-SNE and UMAP visualizations
- Interactive scatter plots with hover insights

### 4. **Single Clustering Website** (`data/websites/clustering_website.html`)
- Explore best clustering result in detail
- Organized cluster insights and file browsing

### 5. **Multi-Clustering Comparison** (`data/websites/multi_clustering_website.html`)
- Compare different clustering approaches side-by-side
- Analyze cluster quality and distribution

## 🔧 Configuration

### Environment Variables (`.env`)
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

### Audio File Organization
Place audio files in the `audio_files/` directory, organized by duration:
- `00_under_10_seconds/` - Very short clips
- `01_10s_to_1min/` - Short voice memos  
- `02_1min_to_5min/` - Medium length memos
- `03_5min_to_20min/` - Longer recordings

## 📊 Pipeline Overview

1. **Transcription**: Convert audio → text using Whisper/Google Speech
2. **Fingerprinting**: Extract semantic insights using Claude
3. **Embeddings**: Generate vector representations using OpenAI
4. **Clustering**: Find semantic groupings using multiple algorithms
5. **Visualization**: Create interactive exploration tools

## 🎨 Visualization Features

- **Dendrogram cutting** at multiple heights
- **t-SNE and UMAP** 2D projections
- **Interactive cluster exploration** with semantic insights
- **Multi-algorithm comparison** for finding optimal clustering
- **Responsive web interfaces** for desktop and mobile

## 🔬 Clustering Algorithms

- **DBSCAN**: Density-based clustering with noise detection
- **Hierarchical**: Tree-based clustering with multiple cut levels
- **Quality scoring**: Automatic evaluation of clustering effectiveness

## 💡 Use Cases

- **Personal insight discovery**: Find patterns in voice journals
- **Research analysis**: Cluster interview or meeting transcripts  
- **Content organization**: Group recordings by semantic themes
- **Knowledge extraction**: Surface key insights from audio data

---

**Hidden Genius** - Unlock the insights hidden in your voice memos! 🎙️✨