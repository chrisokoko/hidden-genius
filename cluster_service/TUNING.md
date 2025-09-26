# Clustering Service Tuning Guide

This guide explains how to tune the clustering service for optimal signal detection across different types of data.

## Understanding the Parameters

The clustering service uses mathematical signal processing to detect natural breaks in clustering quality. Different data types have different characteristics that require parameter adjustments to find the best signal.

### Key Configuration Parameters

#### 1. Peak Detection Sensitivity (`peak_detection`)

Controls how the signal processing detects natural breakpoints:

```python
peak_detection: Dict[str, Optional[float]] = {
    "height": None,        # Minimum quality threshold (None = no minimum)
    "distance": 3.0,       # Minimum separation between peaks
    "prominence": 0.005,   # How much peaks must "stand out"
    "width": 1.0,          # Minimum width for peaks
    "rel_height": 0.5      # Relative height for width measurement
}
```

#### 2. Quality Score Weights (`quality_weights`)

Determines what aspects of clustering quality matter most:

```python
quality_weights: Dict[str, float] = {
    "silhouette": 0.35,    # How well-separated clusters are
    "balance": 0.20,       # How evenly-sized clusters are
    "separation": 0.25,    # Inter vs intra-cluster distances
    "stability": 0.20      # Consistency of clustering
}
```

#### 3. Search Parameters

Controls the clustering search space:

```python
min_clusters: int = 3         # Minimum meaningful clusters
target_count: int = 200       # Number of heights to test
```

## Data Type Characteristics & Tuning Strategies

### Dense, Well-Structured Data
**Examples**: Academic papers, technical documentation, well-organized code

**Characteristics**:
- Clear topical boundaries
- Low noise
- Balanced cluster sizes expected

**Recommended Tuning**:
```python
# config.py adjustments
peak_detection = {
    "height": 0.02,        # Higher quality threshold
    "distance": 5.0,       # Require more separation
    "prominence": 0.01,    # Less sensitive to small changes
    "width": 2.0,          # Wider peaks required
    "rel_height": 0.6
}

quality_weights = {
    "silhouette": 0.30,    # Moderate separation emphasis
    "balance": 0.30,       # Expect balanced clusters
    "separation": 0.25,    # Standard separation
    "stability": 0.15
}

target_count = 150         # Fewer test points needed
```

### Noisy, Conversational Data
**Examples**: Chat messages, social media posts, informal communications

**Characteristics**:
- Rapid topic changes
- High noise levels
- Very unbalanced cluster sizes
- Many small, fragmented groups

**Recommended Tuning**:
```python
# config.py adjustments
peak_detection = {
    "height": None,        # No quality threshold - find all signals
    "distance": 2.0,       # Allow very close peaks
    "prominence": 0.003,   # Very sensitive detection
    "width": 0.5,          # Narrow peaks acceptable
    "rel_height": 0.4
}

quality_weights = {
    "silhouette": 0.30,    # Standard separation
    "balance": 0.10,       # Don't penalize imbalance heavily
    "separation": 0.40,    # High emphasis on separation for noisy data
    "stability": 0.20
}

target_count = 300         # More test points for noisy data
min_clusters = 2           # Allow very few clusters
```

### Code/Technical Data
**Examples**: Programming code, API documentation, technical specifications

**Characteristics**:
- Very distinct functional categories
- Should cluster tightly within categories
- Clear boundaries between different types

**Recommended Tuning**:
```python
# config.py adjustments
peak_detection = {
    "height": 0.05,        # High quality threshold
    "distance": 8.0,       # Require well-separated peaks
    "prominence": 0.02,    # Less sensitive - only clear signals
    "width": 3.0,          # Wide, stable peaks
    "rel_height": 0.7
}

quality_weights = {
    "silhouette": 0.45,    # High emphasis on separation
    "balance": 0.20,       # Moderate balance expectation
    "separation": 0.30,    # High separation importance
    "stability": 0.05      # Less concern about stability
}

target_count = 100         # Fewer test points needed
```

### Mixed/Unknown Data
**Examples**: General purpose embeddings, mixed content types

**Characteristics**:
- Moderate noise
- Mixed cluster patterns
- Unknown structure

**Recommended Tuning** (Current Defaults):
```python
# config.py - these are the current defaults
peak_detection = {
    "height": None,
    "distance": 3.0,
    "prominence": 0.005,
    "width": 1.0,
    "rel_height": 0.5
}

quality_weights = {
    "silhouette": 0.35,
    "balance": 0.20,
    "separation": 0.25,
    "stability": 0.20
}

target_count = 200
min_clusters = 3
```

## How to Apply Custom Tuning

### Method 1: Modify config.py Directly
Edit the default values in `cluster_service/config.py`:

```python
class ClusteringConfig(BaseSettings):
    # Adjust these values based on your data type
    min_clusters: int = 3
    target_count: int = 200

    quality_weights: Dict[str, float] = {
        "silhouette": 0.35,  # Adjust these weights
        "balance": 0.20,
        "separation": 0.25,
        "stability": 0.20
    }

    peak_detection: Dict[str, Optional[float]] = {
        "height": None,      # Adjust these parameters
        "distance": 3.0,
        "prominence": 0.005,
        "width": 1.0,
        "rel_height": 0.5
    }
```

### Method 2: Create Custom Config Programmatically
```python
from cluster_service import ClusteringService, ClusteringConfig

# Create custom config for your data type
custom_config = ClusteringConfig(
    min_clusters=2,
    target_count=300,
    quality_weights={
        "silhouette": 0.30,
        "balance": 0.10,
        "separation": 0.40,
        "stability": 0.20
    },
    peak_detection={
        "height": None,
        "distance": 2.0,
        "prominence": 0.003,
        "width": 0.5,
        "rel_height": 0.4
    }
)

# Use with service
service = ClusteringService(config=custom_config)
result = service.run(embeddings_dir, output_dir)
```

### Method 3: Environment Variables
Set environment variables with the `CLUSTERING_` prefix:

```bash
export CLUSTERING_MIN_CLUSTERS=2
export CLUSTERING_TARGET_COUNT=300
export CLUSTERING_PEAK_DETECTION='{"prominence": 0.003, "distance": 2.0}'
python3 -m cluster_service embeddings/ output/
```

## Iterative Tuning Process

1. **Start with defaults** and run clustering
2. **Analyze results**:
   - Too many small clusters? Decrease sensitivity (increase prominence)
   - Missing subtle groups? Increase sensitivity (decrease prominence)
   - Poor separation? Increase separation weight
   - Unbalanced but that's expected? Decrease balance weight
3. **Adjust one parameter at a time**
4. **Re-run and compare natural breaks found**
5. **Repeat until optimal signal detection**

## Expected Outcomes by Data Type

### Dense/Structured Data
- **Typical clusters found**: 10-100
- **Quality scores**: 0.6-0.8 (high)
- **Natural breaks**: 2-4 clear levels
- **Hierarchy**: Well-defined, stable

### Noisy/Conversational Data
- **Typical clusters found**: 50-500
- **Quality scores**: 0.4-0.6 (moderate)
- **Natural breaks**: 3-6 levels with gradual transitions
- **Hierarchy**: Many small groups with few large ones

### Code/Technical Data
- **Typical clusters found**: 5-50
- **Quality scores**: 0.7-0.9 (very high)
- **Natural breaks**: 2-3 distinct levels
- **Hierarchy**: Sharp, well-defined boundaries

The key is to tune the mathematical signal processing to find the **highest quality natural breaks** that make sense for your specific data characteristics.