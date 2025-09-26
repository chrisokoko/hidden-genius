# Clustering Service

Intelligent hierarchical clustering service that discovers natural groupings in embedding vectors using mathematical signal processing.

## Overview

This service performs sophisticated clustering analysis on embedding vectors, using advanced mathematical techniques to automatically discover the optimal number of clusters and hierarchical levels in your data. Unlike traditional clustering approaches that require you to specify the number of clusters, this service finds natural breakpoints in the data structure using signal processing techniques.

## Key Features

- **🔬 Mathematical Peak Detection**: Uses scipy signal processing to find natural breaks in clustering quality
- **📊 Multi-Signal Analysis**: Analyzes peaks, valleys, and gradient changes in quality metrics
- **🎯 Automatic Hierarchy Discovery**: Finds 1-5 natural hierarchical levels based on data structure
- **⚡ Pure Mathematical Approach**: No arbitrary thresholds or hardcoded cluster counts
- **🏗️ Clean Architecture**: Modular design with separation of concerns

## Quick Start

### Command Line Usage

```bash
python3 -m cluster_service <embeddings_dir> <output_dir> [--verbose]
```

**Example:**
```bash
python3 -m cluster_service data/embeddings data/clusters --verbose
```

### Programmatic Usage

```python
from cluster_service import ClusteringService
from pathlib import Path

service = ClusteringService()
result = service.run(
    embeddings_dir=Path("data/embeddings"),
    output_dir=Path("data/clusters")
)
```

## Input Format

The service expects a directory containing JSON files with embedding vectors:

```json
{
  "embedding_vector": [0.1, 0.2, 0.3, ...],
  "metadata": { ... }
}
```

Each JSON file should contain an `embedding_vector` field with a list of float values.

## Output Format

The service generates a `clustering_results.json` file containing:

```json
{
  "optimal_levels": {
    "coarse": {
      "n_clusters": 6,
      "quality_score": 0.501,
      "clusters": {
        "0": ["file1", "file2"],
        "1": ["file3", "file4"]
      }
    },
    "detailed": {
      "n_clusters": 247,
      "quality_score": 0.655,
      "clusters": { ... }
    }
  },
  "summary": {
    "total_embeddings": 455,
    "natural_levels_found": 5,
    "discovery_method": "mathematical_peak_detection"
  }
}
```

## How It Works

### 1. **Natural Breakpoint Detection**
The service tests ~200 different clustering heights and evaluates quality at each point using multiple metrics:
- **Silhouette Score**: How well-separated clusters are
- **Balance Score**: How evenly-sized clusters are
- **Separation Ratio**: Inter-cluster vs intra-cluster distances

### 2. **Signal Processing Analysis**
Uses scipy's `find_peaks` to detect:
- **Peaks**: Local maxima in clustering quality (optimal cluster counts)
- **Valleys**: Transition points between different granularities
- **Gradients**: Sharp changes indicating natural boundaries

### 3. **Hierarchy Selection**
Automatically selects the most prominent and well-separated natural breaks to create a hierarchical structure with meaningful level names:
- **1 level**: `primary`
- **2 levels**: `coarse`, `fine`
- **3 levels**: `coarse`, `medium`, `fine`
- **4 levels**: `coarse`, `medium`, `fine`, `granular`
- **5+ levels**: `coarse`, `medium`, `fine`, `granular`, `detailed`

## Configuration

Customize clustering behavior via `cluster_service/config.py`:

```python
class ClusteringConfig(BaseSettings):
    # Core parameters
    min_clusters: int = 3
    target_count: int = 200

    # Algorithm settings
    distance_metric: str = "cosine"
    linkage_method: str = "ward"

    # Peak detection sensitivity
    peak_detection: Dict[str, Optional[float]] = {
        "height": None,        # No minimum height requirement
        "distance": 3.0,       # Minimum separation between peaks
        "prominence": 0.005,   # How much peaks must stand out
        "width": 1.0          # Minimum width of peaks
    }
```

## Architecture

The service uses clean, modular architecture:

```
cluster_service/
├── algorithms.py      # Pure mathematical functions
├── io_handler.py      # File I/O operations
├── service.py         # ClusteringService orchestration
├── config.py          # Pydantic configuration
└── cli.py            # Command line interface
```

### Key Classes

- **`ClusteringService`**: Main orchestration class
- **`IOHandler`**: Handles loading/saving of files
- **`ClusteringConfig`**: Pydantic configuration with validation

## Requirements

- Python 3.8+
- numpy
- scipy
- scikit-learn
- pydantic
- pydantic-settings

## Example Results

For a dataset of 455 voice memo embeddings, the service discovered:

- **22 natural breakpoints** using mathematical analysis
- **5 hierarchical levels**:
  - Coarse: 6 clusters (major themes)
  - Medium: 55 clusters (topic areas)
  - Fine: 77 clusters (subtopics)
  - Granular: 170 clusters (specific themes)
  - Detailed: 247 clusters (optimal granularity)

The highest quality clustering (0.655) was found at 247 clusters, representing the natural "sweet spot" for this particular dataset.

## Advanced Usage

### Custom Configuration

```python
from cluster_service import ClusteringService, ClusteringConfig

config = ClusteringConfig(
    min_clusters=5,
    target_count=300,
    peak_detection={
        "prominence": 0.01,  # Less sensitive
        "distance": 5.0      # More separation required
    }
)

service = ClusteringService(config)
result = service.run(embeddings_dir, output_dir)
```

### Error Handling

```python
try:
    service = ClusteringService()
    result = service.run(embeddings_dir, output_dir)
    print(f"Processed {result['processed']} files")
except FileNotFoundError:
    print("Embeddings directory not found")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Mathematical Background

The clustering uses:
- **Ward Linkage**: Minimizes within-cluster variance
- **Cosine Distance**: Measures semantic similarity between embeddings
- **Hierarchical Agglomerative Clustering**: Builds tree structure from bottom-up
- **Signal Processing**: Detects natural boundaries using peak detection algorithms

This approach is particularly effective for high-dimensional semantic embeddings where traditional clustering methods often require manual cluster count specification.

## License

This clustering service is designed for analyzing semantic embeddings and discovering natural groupings in high-dimensional data.