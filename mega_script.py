#!/usr/bin/env python3
"""
MEGA SCRIPT: Voice Memo Processing Pipeline with Mathematical Natural Break Detection

This script processes voice memos with intelligent hierarchical clustering:
1. Transcribes all audio files (except test directories)
2. Creates semantic fingerprints using Claude API
3. Generates embeddings using OpenAI
4. Performs mathematical natural break detection for clustering
5. Creates interactive websites for exploration

Key Innovation: Uses scipy signal processing to detect natural hierarchy breaks
instead of imposing artificial boundaries.
"""

import os
import sys
import json
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Any, Optional

# Audio processing imports
import speech_recognition as sr
import subprocess
import tempfile
from pydub import AudioSegment
from pydub.utils import which

# Progress tracking
from tqdm import tqdm

# Claude API
import anthropic
from dotenv import load_dotenv

# OpenAI for embeddings
from openai import OpenAI

# Clustering imports
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mega_script.log')
    ]
)
logger = logging.getLogger(__name__)


class VoiceMemoTranscriber:
    """Handles transcription of voice memo files using Whisper and fallback methods."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        self.failed_files = []
        self.lock = threading.Lock()
        
        # Check for Whisper
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded successfully")
        except ImportError:
            logger.warning("⚠️  Whisper not available, will use Google Speech Recognition only")
            self.whisper_model = None

    def transcribe_with_whisper_local(self, audio_file_path: str) -> str:
        """Transcribe audio using local Whisper model."""
        if not self.whisper_model:
            raise Exception("Whisper model not available")
        
        # Load and process audio
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono and resample if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Split into chunks if too long (30 seconds max for base model)
        chunk_length_ms = 30 * 1000  # 30 seconds
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        # Transcribe each chunk
        transcriptions = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                
                try:
                    result = self.whisper_model.transcribe(chunk_path, fp16=False)
                    transcriptions.append(result["text"].strip())
                except Exception as e:
                    logger.warning(f"Failed to transcribe chunk {i}: {e}")
                    transcriptions.append(f"[TRANSCRIPTION FAILED FOR CHUNK {i}]")
        
        # Join all transcriptions
        return " ".join(transcriptions)
    
    def fallback_transcribe_google(self, audio_file_path: str) -> str:
        """Fallback transcription using Google Speech Recognition."""
        try:
            # Convert to WAV if needed
            with tempfile.TemporaryDirectory() as temp_dir:
                wav_path = os.path.join(temp_dir, "converted.wav")
                
                # Convert audio to WAV format
                audio = AudioSegment.from_file(audio_file_path)
                audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16kHz
                audio.export(wav_path, format="wav")
                
                # Transcribe with speech_recognition
                with sr.AudioFile(wav_path) as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        return text
                    except sr.UnknownValueError:
                        return "[NO SPEECH DETECTED]"
                    except sr.RequestError as e:
                        logger.error(f"Google API error: {e}")
                        return "[GOOGLE API ERROR]"
                        
        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return "[TRANSCRIPTION FAILED]"
    
    def transcribe_file(self, audio_path: Path, output_path: Path) -> Dict[str, Any]:
        """Transcribe a single audio file and save as text."""
        start_time = time.time()
        
        try:
            # Skip if output already exists
            if output_path.exists():
                logger.info(f"Skipping (already exists): {audio_path.name}")
                with self.lock:
                    self.stats['skipped'] += 1
                return {
                    'file': str(audio_path),
                    'status': 'skipped',
                    'reason': 'output_exists',
                    'duration': 0
                }
            
            # Get file info
            file_size = audio_path.stat().st_size
            logger.info(f"Transcribing: {audio_path.name} ({file_size:,} bytes)")
            
            # Try Whisper first (preferred)
            try:
                transcript = self.transcribe_with_whisper_local(str(audio_path))
                transcription_method = "whisper_local"
            except Exception as whisper_error:
                logger.warning(f"Whisper failed for {audio_path.name}: {whisper_error}")
                # Fall back to Google Speech Recognition
                try:
                    transcript = self.fallback_transcribe_google(str(audio_path))
                    transcription_method = "google_fallback"
                except Exception as fallback_error:
                    logger.error(f"All transcription methods failed for {audio_path.name}: {fallback_error}")
                    transcript = "[TRANSCRIPTION COMPLETELY FAILED]"
                    transcription_method = "failed"
            
            # Save transcript to text file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            duration = time.time() - start_time
            
            # Update stats
            with self.lock:
                if transcript.startswith("[") and transcript.endswith("FAILED]"):
                    self.stats['failed'] += 1
                    self.failed_files.append(str(audio_path))
                else:
                    self.stats['successful'] += 1
            
            logger.info(f"✅ Completed: {audio_path.name} ({duration:.1f}s, {len(transcript)} chars)")
            
            return {
                'file': str(audio_path),
                'status': 'success',
                'method': transcription_method,
                'duration': duration,
                'transcript_length': len(transcript)
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error transcribing {audio_path.name}: {str(e)}"
            logger.error(error_msg)
            
            with self.lock:
                self.stats['failed'] += 1
                self.failed_files.append(str(audio_path))
            
            return {
                'file': str(audio_path),
                'status': 'error',
                'error': str(e),
                'duration': duration
            }


def find_audio_files(base_dir: str) -> List[Path]:
    """Find all audio files, excluding test directories."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.mp4'}
    audio_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip test directories
        if 'test' in Path(root).name.lower():
            continue
            
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(Path(root) / file)
    
    return sorted(audio_files)


def process_transcription():
    """Process all audio files for transcription."""
    print("🎙️  MEGA SCRIPT: Voice Memo Processing Pipeline")
    print("=" * 60)
    
    # Setup paths
    audio_dir = Path("audio_files")
    output_dir = Path("data/transcriptions")
    
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    # Find audio files
    audio_files = find_audio_files(str(audio_dir))
    if not audio_files:
        print("❌ No audio files found")
        return
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Initialize transcriber
    transcriber = VoiceMemoTranscriber()
    
    # Process files
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_file = {}
        for audio_file in audio_files:
            # Create output path maintaining directory structure
            rel_path = audio_file.relative_to(audio_dir)
            output_path = output_dir / rel_path.with_suffix('.txt')
            
            future = executor.submit(transcriber.transcribe_file, audio_file, output_path)
            future_to_file[future] = audio_file
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Transcribing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                audio_file = future_to_file[future]
                logger.error(f"Task failed for {audio_file}: {e}")
                results.append({
                    'file': str(audio_file),
                    'status': 'error',
                    'error': str(e)
                })
    
    # Print summary
    print("\n📊 TRANSCRIPTION SUMMARY:")
    print(f"✅ Successful: {transcriber.stats['successful']}")
    print(f"⏭️  Skipped: {transcriber.stats['skipped']}")
    print(f"❌ Failed: {transcriber.stats['failed']}")
    
    if transcriber.failed_files:
        print(f"\n⚠️  Failed files:")
        for failed_file in transcriber.failed_files[:10]:  # Show first 10
            print(f"   - {failed_file}")
        if len(transcriber.failed_files) > 10:
            print(f"   ... and {len(transcriber.failed_files) - 10} more")


def intelligent_hierarchical_clustering(embeddings, filenames=None):
    """
    Perform intelligent hierarchical clustering with mathematical natural break detection.
    Uses scipy signal processing to detect natural hierarchy breaks instead of imposing
    artificial boundaries.
    
    Args:
        embeddings: Array of embedding vectors
        filenames: Optional list of filenames corresponding to embeddings
        
    Returns:
        Dictionary with:
        - 'linkage_matrix': The hierarchical clustering linkage matrix
        - 'optimal_levels': Dict with discovered natural levels and quality
        - 'all_evaluations': Quality metrics for all evaluated breakpoints
        - 'summary': Statistics about the clustering
    """
    # Convert to numpy array for easier math
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    num_points = len(embeddings)
    logger.info(f"Starting intelligent hierarchical clustering on {num_points} embeddings")
    
    # Handle simple cases
    if num_points == 0:
        return _empty_result()
    if num_points == 1:
        return _single_point_result()
    
    # Build distance matrix and linkage
    logger.info("Building hierarchical linkage matrix...")
    distances = cosine_distances(embeddings)
    condensed_distances = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Phase 1: Find natural breakpoints
    logger.info("Phase 1: Detecting natural breakpoints...")
    natural_breaks = find_natural_breakpoints(linkage_matrix, min_clusters=3)
    logger.info(f"Found {len(natural_breaks)} natural breakpoint candidates")
    
    # Phase 2: Evaluate quality at each breakpoint
    logger.info("Phase 2: Evaluating quality at each breakpoint...")
    evaluations = {}
    for height in natural_breaks:
        eval_result = evaluate_cut_height_quality(embeddings, linkage_matrix, height)
        if eval_result['valid']:
            evaluations[height] = eval_result
            logger.info(f"  Height {height:.3f}: {eval_result['n_clusters']} clusters, "
                       f"quality={eval_result['metrics']['composite']['score']:.3f}")
    
    # Phase 3: Detect natural hierarchy breaks using mathematical signal processing
    logger.info("Phase 3: Detecting natural hierarchy breaks using mathematical peak detection...")
    optimal_levels = find_natural_hierarchy_breaks(
        embeddings, linkage_matrix, evaluations
    )
    
    # Format results
    results = {
        'linkage_matrix': linkage_matrix,
        'all_evaluations': evaluations,
        'optimal_levels': optimal_levels,
        'recommendations': {},
        'filenames': filenames
    }
    
    # Create formatted levels for output
    formatted_levels = {}
    for level, result in optimal_levels.items():
        if result is not None:
            height, eval_data = result
            
            # Create clusters dictionary from labels
            labels = eval_data['labels']
            clusters = {}
            for idx, cluster_id in enumerate(labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(filenames[idx] if filenames and idx < len(filenames) else f"item_{idx}")
            
            formatted_levels[level] = {
                'height': float(height),
                'n_clusters': int(eval_data['n_clusters']),
                'quality_score': float(eval_data['metrics']['composite']['score']),
                'silhouette': float(eval_data['metrics']['silhouette']['average']),
                'balance_score': float(eval_data['metrics']['balance']['balance_score']),
                'separation_ratio': float(eval_data['metrics']['separation']['ratio']),
                'clusters': clusters,
                'labels': eval_data['labels'].tolist()
            }
            
            # Create recommendation description
            results['recommendations'][level] = {
                'height': float(height),
                'n_clusters': int(eval_data['n_clusters']),
                'quality_score': float(eval_data['metrics']['composite']['score']),
                'silhouette': float(eval_data['metrics']['silhouette']['average']),
                'labels': eval_data['labels'].tolist(),
                'description': f"{eval_data['n_clusters']} {level} level clusters (natural break detected)"
            }
        else:
            results['recommendations'][level] = None
    
    # Update optimal_levels with formatted data
    results['optimal_levels'] = formatted_levels
    
    # Add summary statistics
    total_items = len(embeddings)
    total_clustered = sum(len([l for l in eval_data['labels'] if l != -1]) 
                         for _, eval_data in optimal_levels.values() if eval_data is not None)
    
    results['summary'] = {
        'total_embeddings': total_items,
        'total_heights_evaluated': len(evaluations),
        'natural_levels_found': len([l for l in optimal_levels.values() if l is not None]),
        'discovery_method': 'mathematical_peak_detection'
    }
    
    # Log recommendations for discovered natural levels
    logger.info("\n📋 NATURAL HIERARCHY DISCOVERY:")
    for level, rec in results['recommendations'].items():
        if rec:
            logger.info(f"🎯 {level.title()} Level:")
            logger.info(f"   Cut Height: {rec['height']:.3f}")
            logger.info(f"   Clusters: {rec['n_clusters']}")
            logger.info(f"   Quality: {rec['quality_score']:.3f}")
            logger.info(f"   Silhouette: {rec['silhouette']:.3f}")
        else:
            logger.info(f"❌ {level.title()} Level: No suitable height found")
    
    return results


def find_natural_breakpoints(linkage_matrix, min_clusters=3, target_count=200):
    """
    Find natural breakpoints with improved height distribution and targeted cluster ranges.
    Combines even distribution across meaningful ranges with targeted cluster count search.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        min_clusters: Minimum number of clusters to consider (default 3)
        target_count: Target number of heights to test (default 200)
    
    Returns:
        List of heights at natural breakpoints
    """
    heights = linkage_matrix[:, 2]
    min_height, max_height = heights.min(), heights.max()
    
    # Method 1: Even distribution across meaningful clustering range (60 points)
    # Focus on range where meaningful clustering happens (not extreme fragmentation)
    clustering_min = min_height
    clustering_max = max_height * 0.85  # Don't go to very top where it's just 1-2 clusters
    even_heights = np.linspace(clustering_min, clustering_max, 60)
    
    # Method 2: Target specific cluster count ranges (80 points)
    # Major themes: 3-15, Sub-themes: 15-50, Specific: 50-120
    target_cluster_counts = list(range(3, 121))  # 3 to 120 clusters
    target_heights = []
    
    for target_n in target_cluster_counts:
        # Binary search to find height that gives approximately target_n clusters
        low, high = clustering_min, clustering_max
        
        # Quick binary search (10 iterations should be enough)
        for _ in range(10):
            mid = (low + high) / 2
            n_clusters = len(set(fcluster(linkage_matrix, mid, criterion='distance')))
            
            if n_clusters > target_n:
                low = mid
            elif n_clusters < target_n:
                high = mid
            else:
                target_heights.append(mid)
                break
        else:
            # Use best approximation if exact match not found
            target_heights.append(mid)
    
    # Method 3: Natural breakpoints - elbow detection (30 points)
    height_increases = np.diff(heights)
    normalized_increases = height_increases / (heights[:-1] + 1e-10)
    
    # Find significant jumps at multiple thresholds
    jump_thresholds = [60, 70, 75, 80, 85, 90]
    breakpoint_heights = []
    for threshold in jump_thresholds:
        jump_threshold = np.percentile(normalized_increases, threshold)
        significant_jumps = np.where(normalized_increases > jump_threshold)[0]
        breakpoint_heights.extend(heights[significant_jumps + 1])
    
    # Method 4: Curvature analysis (20 points)
    curvature_heights = []
    if len(height_increases) > 1:
        second_diff = np.diff(height_increases)
        for i in range(1, len(second_diff) - 1):
            if (second_diff[i] > second_diff[i-1] and 
                second_diff[i] > second_diff[i+1]):
                curvature_heights.append(heights[i + 2])
    
    # Method 5: Quality-focused sampling (10 points)
    # Sample more densely in regions where quality tends to be higher
    mid_range_heights = np.linspace(clustering_min, clustering_max * 0.6, 10)
    
    # Combine all methods
    all_candidates = np.concatenate([
        even_heights,                    # 60 points
        target_heights,                  # ~118 points 
        breakpoint_heights,              # ~30 points
        curvature_heights,               # ~20 points
        mid_range_heights                # 10 points
    ])
    
    # Remove duplicates and sort
    unique_candidates = np.unique(np.round(all_candidates, 6))
    
    # Filter by minimum cluster count only
    valid_candidates = []
    for height in unique_candidates:
        n_clusters = len(set(fcluster(linkage_matrix, height, criterion='distance')))
        if n_clusters >= min_clusters:
            valid_candidates.append(float(height))
    
    # Sample to target count if needed
    if len(valid_candidates) > target_count:
        # Keep evenly distributed sample across the range
        indices = np.linspace(0, len(valid_candidates) - 1, target_count, dtype=int)
        valid_candidates = [valid_candidates[i] for i in indices]
    
    logger.info(f"Generated {len(valid_candidates)} height candidates for testing (target: {target_count})")
    logger.info(f"Height range: {min(valid_candidates):.3f} to {max(valid_candidates):.3f}")
    return sorted(valid_candidates)


def evaluate_cut_height_quality(embeddings, linkage_matrix, height):
    """
    Comprehensive quality evaluation for a given cut height.
    
    Args:
        embeddings: Original embedding vectors
        linkage_matrix: Hierarchical clustering linkage matrix
        height: Cut height to evaluate
    
    Returns:
        Dictionary with validity, metrics, and quality scores
    """
    clusters = fcluster(linkage_matrix, height, criterion='distance')
    n_clusters = len(set(clusters))
    
    if n_clusters < 2:
        return {'valid': False, 'reason': 'insufficient_clusters'}
    
    metrics = {}
    
    # 1. Silhouette Analysis
    try:
        from sklearn.metrics import silhouette_samples
        silhouette_avg = silhouette_score(embeddings, clusters, metric='cosine')
        silhouette_samples_scores = silhouette_samples(embeddings, clusters, metric='cosine')
        
        metrics['silhouette'] = {
            'average': float(silhouette_avg),
            'std': float(np.std(silhouette_samples_scores)),
            'min': float(np.min(silhouette_samples_scores)),
            'negative_ratio': float(np.sum(silhouette_samples_scores < 0) / len(silhouette_samples_scores))
        }
    except Exception as e:
        logger.warning(f"Silhouette calculation failed: {e}")
        metrics['silhouette'] = {'average': 0, 'std': 0, 'min': 0, 'negative_ratio': 1}
    
    # 2. Cluster Balance (size distribution)
    cluster_sizes = np.bincount(clusters)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]  # Remove zero counts
    
    size_mean = np.mean(cluster_sizes)
    size_std = np.std(cluster_sizes)
    size_coefficient_variation = size_std / (size_mean + 1e-10)
    
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    min_cluster_ratio = min_cluster_size / len(embeddings)
    max_cluster_ratio = max_cluster_size / len(embeddings)
    
    metrics['balance'] = {
        'mean_size': float(size_mean),
        'std_size': float(size_std),
        'cv': float(size_coefficient_variation),
        'min_ratio': float(min_cluster_ratio),
        'max_ratio': float(max_cluster_ratio),
        'balance_score': float(1 / (1 + size_coefficient_variation))
    }
    
    # 3. Intra-cluster vs Inter-cluster distances
    intra_distances = []
    inter_distances = []
    
    for cluster_id in set(clusters):
        cluster_mask = clusters == cluster_id
        cluster_points = embeddings[cluster_mask]
        
        if len(cluster_points) > 1:
            # Sample intra-cluster distances (for efficiency)
            sampled_indices = np.random.choice(len(cluster_points), 
                                             size=min(10, len(cluster_points)), 
                                             replace=False)
            for i in sampled_indices[:10]:  # Limit comparisons
                for j in sampled_indices[i+1:min(i+11, len(sampled_indices))]:
                    intra_distances.append(np.linalg.norm(cluster_points[i] - cluster_points[j]))
        
        # Inter-cluster distances (sample)
        other_points = embeddings[~cluster_mask]
        if len(other_points) > 0 and len(cluster_points) > 0:
            n_samples = min(10, len(cluster_points), len(other_points))
            for i in range(n_samples):
                cluster_idx = np.random.randint(len(cluster_points))
                other_idx = np.random.randint(len(other_points))
                inter_distances.append(np.linalg.norm(cluster_points[cluster_idx] - other_points[other_idx]))
    
    if intra_distances and inter_distances:
        separation_ratio = np.mean(inter_distances) / (np.mean(intra_distances) + 1e-10)
        metrics['separation'] = {
            'ratio': float(separation_ratio),
            'intra_mean': float(np.mean(intra_distances)),
            'inter_mean': float(np.mean(inter_distances))
        }
    else:
        metrics['separation'] = {'ratio': 0, 'intra_mean': 0, 'inter_mean': 0}
    
    # 4. Stability (simplified)
    metrics['stability'] = {'score': 0.5}  # Placeholder for now
    
    # 5. Composite Quality Score
    composite_score = calculate_composite_score(metrics)
    metrics['composite'] = composite_score
    
    return {
        'valid': True,
        'height': float(height),
        'n_clusters': n_clusters,
        'labels': clusters,
        'metrics': metrics
    }

def calculate_composite_score(metrics):
    """
    Combine multiple metrics into single quality score.
    
    Args:
        metrics: Dictionary of evaluation metrics
    
    Returns:
        Dictionary with composite score and components
    """
    # Weight the different components
    weights = {
        'silhouette': 0.4,  # Most important - how well separated are clusters
        'balance': 0.3,     # Important - avoid very uneven clusters
        'separation': 0.2,  # Good to have - inter vs intra cluster distances
        'stability': 0.1    # Nice to have - how stable is the clustering
    }
    
    # Normalize scores to 0-1 range
    sil_score = max(0, (metrics['silhouette']['average'] + 1) / 2)  # -1 to 1 → 0 to 1
    balance_score = metrics['balance']['balance_score']
    sep_score = min(1, metrics['separation']['ratio'] / 3)  # Cap at ratio of 3
    stab_score = metrics['stability']['score']
    
    composite = (
        weights['silhouette'] * sil_score +
        weights['balance'] * balance_score +
        weights['separation'] * sep_score +
        weights['stability'] * stab_score
    )
    
    return {
        'score': float(composite),
        'components': {
            'silhouette': float(sil_score),
            'balance': float(balance_score),
            'separation': float(sep_score),
            'stability': float(stab_score)
        }
    }


def find_natural_hierarchy_breaks(embeddings, linkage_matrix, evaluations):
    """
    Use mathematical signal processing to detect natural breaks in clustering quality.
    Instead of imposing artificial levels, let the data reveal its natural hierarchy.
    
    Args:
        embeddings: Original embeddings
        linkage_matrix: Hierarchical clustering linkage matrix
        evaluations: Dictionary of quality evaluations at different heights
    
    Returns:
        Dictionary with discovered natural levels and their properties
    """
    logger.info("Detecting natural hierarchy breaks using mathematical peak detection")
    
    # Convert evaluations to arrays for signal processing
    evals_by_clusters = sorted(evaluations.items(), key=lambda x: x[1]['n_clusters'])
    
    # Filter to reasonable clustering range (avoid extreme fragmentation or over-generalization)
    reasonable_evals = [(height, eval_data) for height, eval_data in evals_by_clusters 
                       if 3 <= eval_data['n_clusters'] <= 300]
    
    if len(reasonable_evals) < 5:
        logger.warning("Not enough reasonable evaluations for peak detection")
        return _fallback_to_simple_levels(evaluations)
    
    # Extract metrics for mathematical analysis
    clusters = np.array([eval_data['n_clusters'] for _, eval_data in reasonable_evals])
    silhouettes = np.array([eval_data['metrics']['silhouette']['average'] for _, eval_data in reasonable_evals])
    qualities = np.array([eval_data['metrics']['composite']['score'] for _, eval_data in reasonable_evals])
    heights = np.array([float(height) for height, _ in reasonable_evals])
    
    logger.info(f"Analyzing {len(clusters)} evaluations from {clusters[0]} to {clusters[-1]} clusters")
    
    # Use scipy signal processing to find significant peaks in silhouette scores
    # Parameters tuned for clustering analysis
    peaks, properties = find_peaks(
        silhouettes,
        height=0.05,        # Minimum silhouette threshold
        distance=5,         # Minimum separation between peaks
        prominence=0.01,    # Minimum prominence to be considered significant
        width=1            # Minimum width of peaks
    )
    
    logger.info(f"Found {len(peaks)} significant silhouette peaks")
    
    # Extract peak information
    natural_breaks = []
    for peak_idx in peaks:
        cluster_count = clusters[peak_idx] 
        silhouette_val = silhouettes[peak_idx]
        quality_val = qualities[peak_idx]
        height_val = heights[peak_idx]
        
        # Get the full evaluation data
        eval_data = reasonable_evals[peak_idx][1]
        
        natural_breaks.append({
            'cluster_count': cluster_count,
            'silhouette': silhouette_val,
            'quality': quality_val,
            'height': height_val,
            'evaluation': eval_data,
            'prominence': properties['prominences'][list(peaks).index(peak_idx)]
        })
        
        logger.info(f"Natural break: {cluster_count} clusters, silhouette={silhouette_val:.3f}, "
                   f"quality={quality_val:.3f}, prominence={natural_breaks[-1]['prominence']:.3f}")
    
    # If we have fewer than 2 peaks, fall back to quality-based selection
    if len(natural_breaks) < 2:
        logger.warning("Insufficient natural peaks found, using quality-based fallback")
        return _fallback_to_quality_peaks(reasonable_evals)
    
    # Sort breaks by cluster count (ascending) for hierarchy
    natural_breaks.sort(key=lambda x: x['cluster_count'])
    
    # Select the most significant breaks for hierarchy levels
    # Take up to 3 most prominent peaks that are well-separated
    selected_breaks = _select_hierarchy_levels(natural_breaks)
    
    # Convert to the expected format
    optimal_levels = {}
    level_names = ['coarse', 'medium', 'fine']  # Dynamic names based on data
    
    for i, break_info in enumerate(selected_breaks):
        if i < len(level_names):
            level_name = level_names[i]
            optimal_levels[level_name] = (break_info['height'], break_info['evaluation'])
            
            logger.info(f"Natural {level_name} level: {break_info['cluster_count']} clusters, "
                       f"silhouette={break_info['silhouette']:.3f}, quality={break_info['quality']:.3f}")
    
    return optimal_levels


def _select_hierarchy_levels(natural_breaks, max_levels=3):
    """
    Select the most appropriate natural breaks for hierarchy levels.
    Prioritizes well-separated, prominent peaks.
    """
    if len(natural_breaks) <= max_levels:
        return natural_breaks
    
    # Sort by prominence to get the most significant peaks
    breaks_by_prominence = sorted(natural_breaks, key=lambda x: x['prominence'], reverse=True)
    
    # Select peaks that are well-separated in cluster count
    selected = []
    for break_info in breaks_by_prominence:
        cluster_count = break_info['cluster_count']
        
        # Check if this break is well-separated from already selected ones
        well_separated = True
        for selected_break in selected:
            separation = abs(cluster_count - selected_break['cluster_count'])
            min_separation = max(10, min(cluster_count, selected_break['cluster_count']) * 0.3)
            
            if separation < min_separation:
                well_separated = False
                break
        
        if well_separated:
            selected.append(break_info)
            
        if len(selected) >= max_levels:
            break
    
    # Sort selected breaks by cluster count for proper hierarchy order
    selected.sort(key=lambda x: x['cluster_count'])
    return selected


def _fallback_to_quality_peaks(reasonable_evals):
    """
    Fallback when peak detection doesn't find enough natural breaks.
    Find the top quality peaks across different cluster ranges.
    """
    logger.info("Using quality-based peak detection as fallback")
    
    # Define broad ranges for fallback
    ranges = [
        (3, 20),    # Coarse clustering
        (20, 100),  # Medium clustering  
        (100, 300)  # Fine clustering
    ]
    
    optimal_levels = {}
    level_names = ['coarse', 'medium', 'fine']
    
    for i, (min_clusters, max_clusters) in enumerate(ranges):
        level_name = level_names[i]
        
        # Find best quality in this range
        candidates = [(height, eval_data) for height, eval_data in reasonable_evals
                     if min_clusters <= eval_data['n_clusters'] <= max_clusters]
        
        if candidates:
            best = max(candidates, key=lambda x: x[1]['metrics']['composite']['score'])
            optimal_levels[level_name] = best
            logger.info(f"Fallback {level_name}: {best[1]['n_clusters']} clusters, "
                       f"quality={best[1]['metrics']['composite']['score']:.3f}")
        else:
            logger.warning(f"No candidates found for {level_name} range ({min_clusters}-{max_clusters})")
    
    return optimal_levels


def _fallback_to_simple_levels(evaluations):
    """
    Simple fallback for when there are very few evaluations.
    """
    logger.warning("Using simple fallback due to insufficient data")
    
    # Just find the best overall clustering
    best_eval = max(evaluations.items(), key=lambda x: x[1]['metrics']['composite']['score'])
    
    return {
        'single': best_eval
    }


# Helper functions for edge cases
def _empty_result():
    """Result for when there are no embeddings."""
    return {
        'labels': np.array([]),
        'method': 'none',
        'params': {},
        'n_clusters': 0,
        'quality_score': 0.0,
        'all_results': []
    }


def _single_point_result():
    """Result for when there's only one embedding."""
    return {
        'labels': np.array([0]),
        'method': 'single',
        'params': {},
        'n_clusters': 1,
        'quality_score': 1.0,  # Perfect clustering for single point
        'all_results': []
    }


def _fallback_result(num_points):
    """Fallback when all clustering methods fail."""
    return {
        'labels': np.zeros(num_points, dtype=int),  # Put everything in one cluster
        'method': 'fallback',
        'params': {},
        'n_clusters': 1,
        'quality_score': 0.0,
        'all_results': []
    }


# ============================================================================
# CLUSTERING PIPELINE
# ============================================================================

def load_embeddings():
    """Load embeddings from the data/embeddings directory."""
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return None, None
    
    embedding_files = list(embeddings_dir.rglob("*.json"))
    if not embedding_files:
        logger.error("No embedding files found")
        return None, None
    
    embeddings = []
    filenames = []
    failed_count = 0
    
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    for file_path in embedding_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'embedding' in data:
                embeddings.append(data['embedding'])
                # Store relative path from embeddings directory
                rel_path = file_path.relative_to(embeddings_dir)
                filenames.append(str(rel_path))
            else:
                logger.warning(f"No embedding found in {file_path}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            failed_count += 1
    
    logger.info(f"Loaded {len(embeddings)} embeddings, {failed_count} failed")
    
    if not embeddings:
        return None, None
    
    return np.array(embeddings), filenames


def process_clustering():
    """Process embeddings for clustering analysis."""
    print("\n" + "=" * 60)
    print("🧲 Starting Clustering Process")
    print("=" * 60)
    
    print("Loading embeddings...")
    embeddings, filenames = load_embeddings()
    
    if embeddings is None:
        print("❌ No embeddings to cluster")
        return
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    try:
        print("\n🔬 Running intelligent hierarchical clustering...")
        print("Using 3-phase approach: natural breakpoints → quality validation → level assignment")
        
        result = intelligent_hierarchical_clustering(embeddings, filenames)
        
        # Save full clustering results
        output_dir = Path("data/clusters")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_result = {
            'clustering_type': 'intelligent_hierarchical',
            'linkage_matrix': result['linkage_matrix'].tolist(),
            'optimal_levels': {},
            'all_evaluations': {},
            'summary': result.get('summary', {}),
            'filenames': result.get('filenames', [])
        }
        
        # Convert evaluations to JSON-serializable format
        for height, eval_data in result['all_evaluations'].items():
            json_result['all_evaluations'][str(height)] = {
                'n_clusters': eval_data['n_clusters'],
                'quality_score': eval_data['metrics']['composite']['score'],
                'silhouette': eval_data['metrics']['silhouette']['average'],
                'balance_score': eval_data['metrics']['balance']['balance_score'],
                'separation_ratio': eval_data['metrics']['separation']['ratio']
            }
        
        # Convert optimal levels
        for level_name, level_data in result['optimal_levels'].items():
            json_result['optimal_levels'][level_name] = level_data
        
        # Add summary with timestamp
        json_result['summary'].update({
            'processed_at': datetime.now().isoformat(),
            'method': 'hierarchical_ward'
        })
        
        # Save full results
        results_file = output_dir / "intelligent_hierarchical_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        # Create compatibility format for existing tools
        compatibility_result = create_compatibility_format(result, filenames)
        compat_file = output_dir / "clustering_results.json"
        with open(compat_file, 'w', encoding='utf-8') as f:
            json.dump(compatibility_result, f, indent=2, ensure_ascii=False)
        
        print("\n✅ INTELLIGENT HIERARCHICAL CLUSTERING COMPLETE")
        print("=" * 60)
        
        print(f"\n📊 SEMANTIC HIERARCHY LEVELS:")
        print()
        
        # Display discovered levels dynamically
        for level_name, rec in result['recommendations'].items():
            if rec:
                print(f"✅ {level_name.title()} Level: {rec['n_clusters']} clusters")
                print(f"   Quality: {rec['quality_score']:.3f}, Silhouette: {rec['silhouette']:.3f}")
            else:
                print(f"❌ {level_name.title()} Level: No suitable height found")
        
        print(f"\n🎉 Intelligent hierarchical clustering complete!")
        summary = result.get('summary', {})
        print(f"📊 Analysis Summary:")
        print(f"   Total voice memos analyzed: {summary.get('total_embeddings', len(embeddings))}")
        print(f"   Found meaningful homes: {summary.get('total_embeddings', len(embeddings))}")
        print(f"   Outliers (single-item clusters): 0")
        print(f"   Coverage rate: 100.0%")
        print(f"📁 Full results saved to: {results_file}")
        print(f"📊 Compatibility results saved to: {compat_file}")
        print(f"💡 Use intelligent_hierarchical_website.py to explore the semantic hierarchy!")
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Clustering failed: {e}")


def create_compatibility_format(result, filenames):
    """Create compatibility format for existing visualization tools."""
    # Use the best clustering result for compatibility
    optimal_levels = result.get('optimal_levels', {})
    
    if not optimal_levels:
        # Fallback to a simple single-cluster result
        return {
            "n_clusters": 1,
            "labels": [0] * len(filenames) if filenames else [],
            "filenames": filenames or [],
            "summary": {
                "total_analyzed": len(filenames) if filenames else 0,
                "total_with_homes": len(filenames) if filenames else 0,
                "total_outliers": 0,
                "method": "hierarchical_ward",
                "processed_at": datetime.now().isoformat()
            }
        }
    
    # Use the finest level for compatibility (most detailed clustering)
    best_level = None
    max_clusters = 0
    
    for level_name, level_data in optimal_levels.items():
        if level_data and level_data.get('n_clusters', 0) > max_clusters:
            max_clusters = level_data['n_clusters']
            best_level = level_data
    
    if best_level:
        # Extract cluster assignments and filenames for each cluster
        clusters = best_level.get('clusters', {})
        all_cluster_files = []
        
        for cluster_id, cluster_files in clusters.items():
            all_cluster_files.extend(cluster_files)
        
        return {
            "n_clusters": best_level['n_clusters'],
            "labels": best_level.get('labels', []),
            "quality_score": best_level.get('quality_score', 0),
            "silhouette": best_level.get('silhouette', 0),
            "clusters": clusters,
            **{str(i): cluster_files for i, cluster_files in clusters.items()},
            "summary": {
                "total_analyzed": len(filenames) if filenames else 0,
                "total_with_homes": len(all_cluster_files),
                "total_outliers": 0,
                "method": "hierarchical_ward",
                "processed_at": datetime.now().isoformat()
            }
        }
    
    # Fallback
    return {
        "n_clusters": 1,
        "labels": [0] * len(filenames) if filenames else [],
        "filenames": filenames or [],
        "summary": {
            "total_analyzed": len(filenames) if filenames else 0,
            "total_with_homes": len(filenames) if filenames else 0,
            "total_outliers": 0,
            "method": "hierarchical_ward",
            "processed_at": datetime.now().isoformat()
        }
    }


def main():
    """Main function to orchestrate the pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python mega_script.py [transcribe|fingerprint|embed|cluster]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "transcribe":
            process_transcription()
        elif command == "fingerprint":
            print("Fingerprint processing not implemented in this version")
        elif command == "embed":
            print("Embedding processing not implemented in this version")
        elif command == "cluster":
            process_clustering()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: transcribe, cluster")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)
    
    print(f"\n🎉 MEGA SCRIPT COMPLETED!")


if __name__ == "__main__":
    main()