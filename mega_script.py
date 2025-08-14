#!/usr/bin/env python3
"""
MEGA SCRIPT: Voice Memo Processing Pipeline

This script processes all voice memos in the audio_files directory:
1. Transcribes all audio files (except test directories)
2. Saves transcriptions as text files

Processes ~533 audio files across 4 duration categories.
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
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available. Install with: pip install umap-learn")

# Load environment variables
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mega_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Semantic Analysis Prompt
SEMANTIC_PROMPT = """You are an expert at analyzing insights and extracting their essential cognitive DNA. Your task is to read a text document and extract its semantic fingerprint according to the comprehensive schema below.

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, or commentary before or after the JSON. Your entire response must be a valid JSON array containing the semantic fingerprints.

## Input

A text document (may contain filler words, incomplete sentences, stream-of-consciousness thinking)

## Output Format

Return ONLY a JSON array with semantic fingerprint objects. Each object follows this structure:

```json
{
  "core_exploration": {
    "central_question": "The fundamental inquiry driving this insight",
    "key_tension": "The productive contradiction being reconciled",
    "breakthrough_moment": "The specific realization that shifted understanding",
    "edge_of_understanding": "What they recognize they haven't figured out yet"
  },

  "conceptual_dna": [
    "2-4 essential concept-patterns that capture core wisdom",
    "Should be quotable standalone insights"
  ],

  "insight_pattern": {
    "thinking_styles": ["array of thinking styles from taxonomy"],
    "insight_type": "single category from taxonomy",
    "development_stage": "single maturity level from taxonomy",
    "connected_domains": ["array of domains from taxonomy"]

  },

  "insight_quality": {
    "uniqueness_score": 0.0,
    "depth_score": 0.0,
    "generative_score": 0.0,
    "usefulness_score": 0.0,
    "confidence_score": 0.0
  },

  "raw_essence": "2-3 sentences capturing core insight in natural language"
}

```

## Schema Definition & Scoring Rubrics

## Core Exploration

- Central Question (Semantic)
    
    **Definition:** The fundamental inquiry driving this insight - the deeper question beneath the surface topic that the person is genuinely trying to resolve, even if they don't explicitly state it as a question.
    
    **Identification Markers:**
    
    - Often implicit but can be inferred from what they're exploring
    - The "why" behind their observation or experience
    - The problem they're trying to solve through this thinking
    - The understanding they're seeking to develop
    
    **Examples:**
    
    - "How do we create deep connection without losing ourselves?"
    - "What enables transformation in stuck systems?"
    - "Why do some practices create lasting change while others don't?"
    - "How can structure and freedom coexist?"
- Key Tension (Semantic)
    
    **Definition:** The productive contradiction or paradox at the heart of the insight. This is the "both/and" that the person is trying to reconcile - two seemingly opposing forces that both feel true.
    
    **Identification Markers:**
    
    - Two concepts that seem mutually exclusive but both necessary
    - The "versus" that's actually an "and"
    - The polarity that generates creative energy
    - Often the source of the breakthrough
    
    **Examples:**
    
    - "Structure vs. emergence"
    - "Solitude vs. connection"
    - "Efficiency vs. humanity"
    - "Boundaries vs. intimacy"
- Breakthrough Moment (Semantic)
    
    **Definition:** The specific realization that shifted understanding - the moment when something clicked, revealed itself, or suddenly made sense. This is the "aha" that transforms confusion into clarity or reveals a new way of seeing.
    
    **Identification Markers:**
    
    - Sudden clarity after confusion or struggle
    - A reversal of previous understanding
    - Discovery that two things thought separate are actually connected
    - The moment when a pattern becomes visible
    - Often accompanied by surprise or excitement
    
    **Examples:**
    
    - "Yielding creates more connection than pursuing"
    - "The resistance IS the information"
    - "Structure doesn't limit creativity - it enables it"
    - "My personal patterns mirror organizational patterns"
- Edge of Understanding (Semantic)
    
    **Definition:** The frontier where clarity becomes uncertain - what the person recognizes they haven't figured out yet. This is where their current insight reaches its limit and points toward the next exploration needed.
    
    **Identification Markers:**
    
    - Questions without answers at the end of the insight
    - Acknowledged limitations of current understanding
    - Recognition that something works in one context but uncertain about others
    - The "but I don't know why" or "but I can't explain how"
    - Where confidence drops and speculation begins
    
    **Examples:**
    
    - "This works one-on-one, but I don't know how to scale it"
    - "I see the pattern but can't explain why it works"
    - "There's something here about timing I haven't figured out"
    - "This applies to relationships, but does it work in organizations?"
- Conceptual DNA (Semantic)
    
    **Definition:** The 2-4 essential concept-patterns that capture the core wisdom of this insight. These are not just topics or categories, but insight phrases that could stand alone as valuable understanding.
    
    **Identification Markers:**
    
    - Could be quoted as standalone wisdom
    - Captures a relationship or dynamic, not just a topic
    - The "quotable" essence of the insight
    
    **Examples:**
    
    - "Embodied awareness as relational bridge"
    - "Yielding as pathway to intimacy"
    - "Constraints as creative catalysts"
    - "Vulnerability as leadership strength"

## Insight Pattern

- Insight Types (Semantic)
    
    **Definition:** The primary category of intellectual contribution this insight represents - what kind of knowledge artifact is being developed through this thinking. 
    
    **Returns:** Returns a string with a single value from the taxonomy.
    
    **Taxonomy:**
    
    - **`observation`** - A noticed pattern or phenomenon captured without interpretation or analysis. Raw experiential data that caught attention but hasn't been processed into deeper understanding.
    - **`methodology`** - A reproducible approach or systematic process that others could follow to achieve similar results. Codified knowledge refined into teachable, transferable steps.
    - **`framework`** - A structural model for organizing understanding within a domain. Conceptual architecture that identifies key components, relationships, and organizing principles.
    - **`philosophy`** - A worldview or fundamental principle about the nature of reality. Core convictions and assumptions that underlie other forms of thinking and guide interpretation of experience.
    - **`synthesis`** - A novel combination bringing together previously separate ideas to create new understanding. Creative integration where existing elements merge to reveal unexpected connections.
    - **`theory`** - An explanatory model with causal mechanisms and predictive power. Systematic understanding of why phenomena occur, offering testable explanations that can anticipate outcomes.
    - **`question`** - Articulation of unknowing that opens new territory for exploration. Generative inquiry that reveals the edges of current understanding and creates space for discovery.
    - **`distinction`** - Recognition of an important difference previously unseen or conflated. Discriminating between concepts that appear similar but operate according to different principles.
    - **`hypothesis`** - A tentative explanation bridging observation and theory, inviting testing. An educated guess about mechanisms or relationships while acknowledging uncertainty.
    - **`practice`** - A specific action or embodied approach being developed through repeated engagement. Implementation-focused insight emphasizing experiential learning and personal discipline.
    - **`story`** - Narrative sense-making that constructs meaning and identity from experience. Personal mythology that creates coherence and significance through interpretive frameworks.
- Thinking Styles (Semantic)
    
    **Definition:** The characteristic cognitive approach(es) used in this insight - *how* the person is processing information and generating understanding, not *what* they're thinking about. 
    
    **Returns:** Returns an array that may have multiple values from the taxonomy: `["embodied-somatic", "systems-holistic"]` for someone using body awareness to understand systems
    
    **Taxonomy** 
    
    - **`analytical-linear`** - Sequential logical reasoning that breaks complex problems into component parts, establishing clear cause-and-effect relationships through step-by-step deduction.
    - **`mathematical-formal`** - Abstract reasoning through symbols, equations, and formal logical systems where relationships are expressed through mathematical structures rather than natural language.
    - **`systems-holistic`** - Pattern recognition that emphasizes interconnections, feedback loops, and emergent properties arising from whole-system dynamics rather than individual components.
    - **`embodied-somatic`** - Knowledge processing through bodily sensation, physical awareness, and felt sense, where the body serves as both source and validator of understanding.
    - **`narrative-temporal`** - Meaning-making through story structure, chronological sequence, and developmental arcs that create coherence through time and relationship.
    - **`metaphorical-associative`** - Understanding through analogy, similarity patterns, and cross-domain mapping where insights emerge from recognizing structural resemblances.
    - **`dialectical-synthetic`** - Integration of apparent contradictions by finding higher-order synthesis that transcends and includes opposing perspectives or forces.
    - **`intuitive-emergent`** - Direct knowing that arises spontaneously without deliberate analysis, trusting non-rational sources of understanding and allowing insights to surface naturally.
    - **`contemplative-receptive`** - Understanding that emerges through sustained attention, witnessing awareness, and receptive presence rather than active analysis or manipulation.
    - **`experimental-iterative`** - Knowledge development through systematic testing, adjustment based on results, and progressive refinement through empirical engagement.

- Development Stage (Semantic)
    
    **Definition:** The most relevant development stage for this insight in the intellectual journey.
    
    **Returns:** Returns a string with a single value from the taxonomy.
    
    **Taxonomy:**
    
    - **`noticing`** - First awareness of something worth attention before any conceptual understanding emerges. Pre-conceptual recognition where curiosity is sparked but no interpretation or analysis has yet begun.
    - **`exploring`** - Active investigation through questions and observations, gathering experiential data without yet forming coherent understanding. The insight involves uncertainty and inquiry-based engagement.
    - **`developing`** - Building understanding through pattern recognition and conceptual connections, where hypotheses form and relationships between ideas become visible. Active sense-making and framework construction.
    - **`breakthrough`** - Sudden shift in understanding where previously unclear patterns suddenly crystallize into coherent insight. Quantum leap in comprehension that fundamentally changes perspective.
    - **`integrating`** - Connecting new understanding to existing knowledge frameworks, weaving the insight into broader conceptual architecture and established mental models.
    - **`refining`** - Polishing and clarifying understanding through precision work, sharpening distinctions and definitions to achieve greater accuracy and nuance.
    - **`applying`** - Testing understanding through practical implementation, moving from theoretical comprehension to real-world engagement and empirical validation.
    - **`teaching`** - Understanding mature enough for transmission to others, with sufficient clarity and completeness to guide someone else's learning and development.
- Connected Domains (Semantic)
    
    **Definition:** The different areas of life, work, or knowledge that this insight links together - identifying when someone is applying understanding from one domain to another.
    
    **Taxonomy:**
    
    ### Personal Domains:
    
    - `personal_practice` - Meditation, exercise, journaling, self-care
    - `relationships` - Intimate partnerships, family, friendships
    - `health_wellness` - Physical health, mental health, healing
    - `spirituality` - Consciousness, awareness, transcendence, meaning
    
    ### Professional Domains:
    
    - `business_strategy` - Planning, competition, market dynamics
    - `leadership` - Team management, vision, influence
    - `organizational` - Culture, systems, structure, process
    - `career` - Professional development, skills, advancement
    
    ### Creative Domains:
    
    - `artistic` - Visual art, music, writing, performance
    - `design` - UX, architecture, product design
    - `innovation` - Invention, R&D, breakthrough thinking
    
    ### Intellectual Domains:
    
    - `systems_thinking` - Complexity, emergence, feedback loops
    - `psychology` - Mind, behavior, cognition, emotion
    - `philosophy` - Ethics, metaphysics, epistemology
    - `science` - Physics, biology, chemistry, research
    
    ### Social Domains:
    
    - `community` - Collective, group dynamics, belonging
    - `education` - Learning, teaching, pedagogy
    - `social_change` - Activism, justice, transformation
    - `culture` - Traditions, values, collective meaning

## Insight Quality

- Uniqueness Score (Semantic)
    
    **Definition:** How distinctive or rare this perspective is compared to conventional thinking - measuring whether this insight represents a common observation or a revolutionary viewpoint.
    
    **Score Ranges:**
    
    - `0.0-0.2` - Common knowledge, widely understood
    - `0.3-0.4` - Somewhat unique angle on known concept
    - `0.5-0.6` - Distinctive perspective, uncommon connection
    - `0.7-0.8` - Rare insight, novel approach
    - `0.9-1.0` - Revolutionary, paradigm-shifting perspective
- Depth Score (Semantic)
    
    **Definition:** How far below surface observation this insight penetrates - whether it addresses symptoms or root causes, patterns or principles, what happens or why it happens.
    
    **Depth Levels:**
    
    - `0.0-0.2` - Surface observation: "I noticed X"
    - `0.3-0.4` - Pattern recognition: "X happens when Y"
    - `0.5-0.6` - Mechanism understanding: "X happens because Z"
    - `0.7-0.8` - Principle articulation: "This reflects fundamental law"
    - `0.9-1.0` - Meta-principle: "The principle itself follows pattern"
- Generative Score (Semantic)
    
    **Definition:** The likelihood that this insight will spawn additional insights - whether it opens new territories, raises new questions, or provides a lens for reexamining other experiences.
    
    **Potential Levels:**
    
    - `0.0-0.2` - Dead end, complete in itself
    - `0.3-0.4` - Minor extensions possible
    - `0.5-0.6` - Several follow-up questions raised
    - `0.7-0.8` - Opens new territory for exploration
    - `0.9-1.0` - Paradigm shift enabling many new insights
- Usefulness Score (Semantic)
    
    **Definition:** How practically applicable this insight is for solving real-world problems or improving lived experience - measuring the potential for concrete implementation and beneficial outcomes.
    
    **Usefulness Levels:**
    
    - `0.0-0.2` - Abstract or theoretical with no clear practical application
    - `0.3-0.4` - Potential utility but requires significant adaptation or development
    - `0.5-0.6` - Clear practical value with moderate implementation barriers
    - `0.7-0.8` - Highly actionable with direct applicability to real problems
    - `0.9-1.0` - Immediately implementable with transformative practical impact
- Confidence Score (Semantic)
    
    **Definition:** The degree of certainty expressed in this insight - how sure the person sounds about their understanding, ranging from tentative exploration to absolute conviction.
    
    **Scale:**
    
    - `0.0-0.2` - Very uncertain: "Maybe", "possibly", "might be"
    - `0.3-0.4` - Tentative: "It seems like", "appears to be"
    - `0.5-0.6` - Moderate: "I think", "probably"
    - `0.7-0.8` - Confident: "I believe", "clearly"
    - `0.9-1.0` - Certain: "Definitely", "absolutely", "I know"

## Additional Fields

- Raw Essence (Semantic)
    
    **Definition:** The core insight captured in 2-3 natural sentences, preserving the original voice and energy of the realization.
    
    **Purpose:** Human-readable summary that maintains authenticity
    
    **Example:** "Meditation with Sarah created boundary dissolution. This yielding mirrors the dance workshop. Key insight: personal practice patterns might be scalable to organizational change."
    

---

**Text Document:**"""

class MegaAudioTranscriber:
    """Enhanced audio transcriber for batch processing."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Setup ffmpeg for pydub
        AudioSegment.converter = which("ffmpeg")
        AudioSegment.ffmpeg = which("ffmpeg")
        AudioSegment.ffprobe = which("ffprobe")
        
        # Stats tracking
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        self.failed_files = []
        self.lock = threading.Lock()
        
    def transcribe_with_whisper_local(self, audio_file_path: str) -> str:
        """
        Use OpenAI Whisper locally with chunking for long files.
        """
        try:
            import whisper
            
            # Load Whisper model (base model - good balance of speed/accuracy)
            model = whisper.load_model("base")
            
            # Check audio duration first
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                duration_minutes = len(audio_segment) / (1000 * 60)
                logger.debug(f"Audio duration: {duration_minutes:.1f} minutes - {audio_file_path}")
                
                # If audio is longer than 15 minutes, chunk it
                if duration_minutes > 15:
                    logger.info(f"Long audio detected ({duration_minutes:.1f}m), using chunking: {Path(audio_file_path).name}")
                    return self._transcribe_long_audio_chunked(model, audio_segment)
                else:
                    # For shorter audio, transcribe directly
                    result = model.transcribe(audio_file_path, fp16=False)
                    return result["text"].strip()
                    
            except Exception as e:
                logger.warning(f"Could not determine duration, trying direct transcription: {e}")
                result = model.transcribe(audio_file_path, fp16=False)
                return result["text"].strip()
                
        except ImportError:
            logger.error("OpenAI Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_file_path}: {e}")
            raise
    
    def _transcribe_long_audio_chunked(self, model, audio_segment: AudioSegment) -> str:
        """Transcribe long audio by chunking into smaller pieces."""
        chunk_length_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
        chunks = []
        
        # Split audio into chunks
        for i in range(0, len(audio_segment), chunk_length_ms):
            chunk = audio_segment[i:i + chunk_length_ms]
            chunks.append(chunk)
        
        # Transcribe each chunk
        transcriptions = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                
                try:
                    result = model.transcribe(chunk_path, fp16=False)
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
                'transcript_length': len(transcript),
                'duration': duration,
                'file_size': file_size
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error transcribing {audio_path.name}: {e}"
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

def collect_audio_files() -> List[Path]:
    """Collect all audio files to process (excluding test directories)."""
    base_dir = Path("audio_files")
    audio_files = []
    
    # Process each duration category
    categories = [
        "00_under_10_seconds",
        "01_10s_to_1min", 
        "02_1min_to_5min",
        "03_5min_to_20min",
        "04_over_20_minutes"
    ]
    
    for category in categories:
        category_dir = base_dir / category
        if category_dir.exists():
            # Find all .m4a files in this category
            m4a_files = list(category_dir.glob("*.m4a"))
            audio_files.extend(m4a_files)
            logger.info(f"Found {len(m4a_files)} files in {category}")
    
    logger.info(f"Total audio files to process: {len(audio_files)}")
    return audio_files

def create_output_structure() -> Path:
    """Create output directory structure for transcripts."""
    output_dir = Path("data/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each category
    categories = [
        "00_under_10_seconds",
        "01_10s_to_1min",
        "02_1min_to_5min", 
        "03_5min_to_20min",
        "04_over_20_minutes"
    ]
    
    for category in categories:
        (output_dir / category).mkdir(exist_ok=True)
    
    return output_dir

def get_output_path(audio_path: Path, output_dir: Path) -> Path:
    """Get the output text file path for an audio file."""
    # Extract category from the path
    category = audio_path.parent.name
    
    # Create output filename (replace .m4a with .txt)
    output_filename = audio_path.stem + ".txt"
    
    return output_dir / category / output_filename

def process_all_files(max_workers: int = 4) -> Dict[str, Any]:
    """Process all audio files with parallel transcription."""
    
    # Initialize
    transcriber = MegaAudioTranscriber()
    audio_files = collect_audio_files()
    output_dir = create_output_structure()
    
    transcriber.stats['total_files'] = len(audio_files)
    transcriber.stats['start_time'] = datetime.now()
    
    logger.info(f"🚀 Starting transcription of {len(audio_files)} files with {max_workers} workers")
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(
                transcriber.transcribe_file,
                audio_path,
                get_output_path(audio_path, output_dir)
            ): audio_path
            for audio_path in audio_files
        }
        
        # Process completed jobs with progress bar
        with tqdm(total=len(audio_files), desc="Transcribing", unit="files") as pbar:
            for future in as_completed(future_to_file):
                audio_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Future failed for {audio_path}: {e}")
                    results.append({
                        'file': str(audio_path),
                        'status': 'future_error',
                        'error': str(e)
                    })
                finally:
                    pbar.update(1)
    
    # Final stats
    transcriber.stats['end_time'] = datetime.now()
    total_duration = (transcriber.stats['end_time'] - transcriber.stats['start_time']).total_seconds()
    
    logger.info("🎉 Transcription completed!")
    logger.info(f"📊 Final Stats:")
    logger.info(f"   Total files: {transcriber.stats['total_files']}")
    logger.info(f"   Successful: {transcriber.stats['successful']}")
    logger.info(f"   Failed: {transcriber.stats['failed']}")
    logger.info(f"   Skipped: {transcriber.stats['skipped']}")
    logger.info(f"   Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    if transcriber.failed_files:
        logger.warning(f"❌ Failed files ({len(transcriber.failed_files)}):")
        for failed_file in transcriber.failed_files[:10]:  # Show first 10
            logger.warning(f"   - {failed_file}")
        if len(transcriber.failed_files) > 10:
            logger.warning(f"   ... and {len(transcriber.failed_files) - 10} more")
    
    # Save results summary
    summary = {
        'stats': transcriber.stats,
        'failed_files': transcriber.failed_files,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = Path("data/transcription_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📝 Summary saved to {summary_path}")
    return summary

def process_semantic_fingerprints():
    """Process transcripts in batches through Claude for semantic fingerprinting."""
    print("🧬 Starting Batch Semantic Fingerprint Processing")
    print("=" * 60)
    
    # Get Claude API key
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        logger.error("CLAUDE_API_KEY not found in .env file!")
        return 1
    
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Collect all transcript files that need processing
    transcript_dir = Path("data/transcripts")
    output_dir = Path("data/fingerprints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files that need processing
    pending_files = []
    for transcript_path in transcript_dir.rglob("*.txt"):
        relative_path = transcript_path.relative_to(transcript_dir)
        output_path = output_dir / relative_path.with_suffix('.json')
        
        if not output_path.exists():
            # Check if file has content
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if len(content) >= 20:
                    pending_files.append(transcript_path)
            except Exception:
                pass
    
    logger.info(f"Found {len(pending_files)} files needing processing")
    
    # Process in batches of 10
    batch_size = 10
    total_batches = (len(pending_files) + batch_size - 1) // batch_size
    processed = 0
    failed = 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(pending_files))
        batch_files = pending_files[start_idx:end_idx]
        
        print(f"\n📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)")
        
        # Build batch prompt
        batch_docs = []
        file_paths = []
        
        for i, file_path in enumerate(batch_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                relative_path = file_path.relative_to(transcript_dir)
                batch_docs.append(f"DOCUMENT {i+1} ({relative_path}):\n{content}")
                file_paths.append(file_path)
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                failed += 1
        
        if not batch_docs:
            continue
        
        # Create batch prompt
        batch_prompt = f"""Process these {len(batch_docs)} transcripts and return a JSON array with {len(batch_docs)} semantic fingerprints.

{SEMANTIC_PROMPT.replace('**Text Document:**', '')}

Return a JSON array where each element is a complete fingerprint object:

"""
        
        for doc in batch_docs:
            batch_prompt += f"\n---{doc}---\n"
        
        try:
            logger.info(f"Sending batch to Claude...")
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                temperature=0.3,
                messages=[{"role": "user", "content": batch_prompt}]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON array
            try:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    fingerprints = json.loads(json_str)
                    logger.info(f"Got {len(fingerprints)} fingerprints from batch")
                else:
                    raise ValueError("No JSON array found")
                
                # Save each fingerprint
                for i, fingerprint in enumerate(fingerprints):
                    if i < len(file_paths):
                        file_path = file_paths[i]
                        relative_path = file_path.relative_to(transcript_dir)
                        output_path = output_dir / relative_path.with_suffix('.json')
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Add metadata
                        fingerprint['metadata'] = {
                            'source_file': str(relative_path),
                            'processed_at': datetime.now().isoformat(),
                            'batch_num': batch_num + 1,
                            'model': 'claude-3-5-sonnet-20241022'
                        }
                        
                        # Save
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(fingerprint, f, indent=2)
                        
                        logger.info(f"✅ Saved: {relative_path}")
                        processed += 1
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON parse error for batch {batch_num + 1}: {e}")
                logger.error(f"Response sample: {response_text[:500]}...")
                failed += len(file_paths)
            
            # Rate limit between batches
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"API error for batch {batch_num + 1}: {e}")
            failed += len(file_paths)
    
    print(f"\n✅ Processed: {processed}")
    print(f"❌ Failed: {failed}")
    return 0

def process_embeddings():
    """Process fingerprints to generate OpenAI embeddings."""
    print("🔮 Starting Embedding Processing")
    print("=" * 60)
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment!")
        return 1
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)
    
    # Collect all fingerprint files
    fingerprints_dir = Path("data/fingerprints")
    fingerprint_files = list(fingerprints_dir.rglob("*.json"))
    logger.info(f"Found {len(fingerprint_files)} fingerprint files")
    
    # Create output directory
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    failed = 0
    skipped = 0
    
    for fingerprint_path in fingerprint_files:
        try:
            # Read fingerprint
            with open(fingerprint_path, 'r', encoding='utf-8') as f:
                fingerprint = json.load(f)
            
            # Create output path - same structure as fingerprint
            relative_path = fingerprint_path.relative_to(fingerprints_dir)
            output_path = output_dir / relative_path
            
            # Skip if already exists
            if output_path.exists():
                logger.info(f"Skipping existing: {relative_path}")
                skipped += 1
                continue
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract text for embedding
            embedding_text_parts = []
            
            # Core exploration fields
            core_exploration = fingerprint.get('core_exploration', {})
            if core_exploration.get('central_question'):
                embedding_text_parts.append(f"Central Question: {core_exploration['central_question']}")
            if core_exploration.get('key_tension'):
                embedding_text_parts.append(f"Key Tension: {core_exploration['key_tension']}")
            if core_exploration.get('breakthrough_moment'):
                embedding_text_parts.append(f"Breakthrough: {core_exploration['breakthrough_moment']}")
            if core_exploration.get('edge_of_understanding'):
                embedding_text_parts.append(f"Edge: {core_exploration['edge_of_understanding']}")
            
            # Conceptual DNA
            conceptual_dna = fingerprint.get('conceptual_dna', [])
            if conceptual_dna:
                dna_text = " ".join(conceptual_dna)
                embedding_text_parts.append(f"Conceptual DNA: {dna_text}")
            
            # Raw essence
            raw_essence = fingerprint.get('raw_essence', '')
            if raw_essence:
                embedding_text_parts.append(f"Raw Essence: {raw_essence}")
            
            # Combine all parts
            embedding_text = " | ".join(embedding_text_parts)
            
            if not embedding_text:
                logger.warning(f"No embedding text extracted from: {relative_path}")
                failed += 1
                continue
            
            logger.info(f"Processing embedding: {relative_path}")
            
            # Generate embedding using OpenAI
            response = openai_client.embeddings.create(
                input=embedding_text,
                model="text-embedding-3-small"
            )
            
            embedding_vector = response.data[0].embedding
            
            # Create embedding data structure
            embedding_data = {
                "embedding_vector": embedding_vector,
                "embedding_text": embedding_text,
                "model": "text-embedding-3-small",
                "dimensions": len(embedding_vector),
                "source_fingerprint": str(relative_path),
                "processed_at": datetime.now().isoformat()
            }
            
            # Save embedding
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, indent=2)
            
            logger.info(f"✅ Saved embedding: {relative_path}")
            processed += 1
            
            # Rate limit
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing {fingerprint_path}: {e}")
            failed += 1
    
    print(f"\n✅ Processed: {processed}")
    print(f"⏭️ Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    return 0

# ============================================================================
# CLUSTERING FUNCTIONS - INTELLIGENT HIERARCHICAL ONLY
# ============================================================================

def intelligent_hierarchical_clustering(embeddings, filenames=None):
    """
    Perform intelligent hierarchical clustering with natural breakpoint detection.
    Uses a three-phase approach:
    1. Detect natural structural breakpoints
    2. Validate quality at each breakpoint
    3. Select optimal heights for major themes, sub-themes, and specific topics
    
    Args:
        embeddings: Array of embedding vectors
        filenames: Optional list of filenames corresponding to embeddings
        
    Returns:
        Dictionary with:
        - 'linkage_matrix': The hierarchical clustering linkage matrix
        - 'optimal_levels': Dict with 'major', 'sub', 'specific' cut heights and quality
        - 'all_evaluations': Quality metrics for all evaluated breakpoints
        - 'recommendations': Formatted recommendations for each level
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
    
    # Build linkage matrix
    logger.info("Building hierarchical linkage matrix...")
    distance_matrix = cosine_distances(embeddings)
    condensed_distances = squareform(distance_matrix)
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
        metrics['silhouette'] = {'average': -1, 'std': 1, 'min': -1, 'negative_ratio': 1}
    
    # 2. Cluster Balance
    cluster_sizes = [np.sum(clusters == i) for i in set(clusters)]
    size_coefficient_variation = np.std(cluster_sizes) / np.mean(cluster_sizes)
    min_cluster_ratio = min(cluster_sizes) / len(clusters)
    max_cluster_ratio = max(cluster_sizes) / len(clusters)
    
    metrics['balance'] = {
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
            # Sample distances for efficiency
            n_samples = min(len(cluster_points), 50)
            sampled_indices = np.random.choice(len(cluster_points), n_samples, replace=False)
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
        'silhouette': 0.35,
        'balance': 0.20,
        'separation': 0.25,
        'stability': 0.20
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
            
            # Apply fragmentation penalty ONLY to fine level
            if level_name == 'fine':
                # Calculate fragmentation penalty to reduce excessive small clusters
                optimal_levels[level_name] = _select_fine_level_with_fragmentation_penalty(
                    natural_breaks, reasonable_evals, embeddings, linkage_matrix
                )
                if optimal_levels[level_name]:
                    height, eval_data = optimal_levels[level_name]
                    logger.info(f"Fine level (with fragmentation penalty): {eval_data['n_clusters']} clusters, "
                               f"quality={eval_data['metrics']['composite']['score']:.3f}")
            else:
                # Use original mathematical approach for coarse/medium
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


def _select_fine_level_with_fragmentation_penalty(natural_breaks, reasonable_evals, embeddings, linkage_matrix):
    """
    Select fine level with fragmentation penalty to reduce excessive small clusters.
    Evaluates candidates in the fine clustering range and applies penalty for high fragmentation.
    """
    logger.info("Applying fragmentation penalty for fine level selection")
    
    # Define fine clustering range (more than medium, but not excessive)
    fine_min_clusters = 50
    fine_max_clusters = 300
    
    # Get all candidates in the fine range
    fine_candidates = [(height, eval_data) for height, eval_data in reasonable_evals
                      if fine_min_clusters <= eval_data['n_clusters'] <= fine_max_clusters]
    
    if not fine_candidates:
        logger.warning("No fine level candidates found, using fallback")
        return None
    
    best_candidate = None
    best_adjusted_score = -1
    
    for height, eval_data in fine_candidates:
        original_score = eval_data['metrics']['composite']['score']
        n_clusters = eval_data['n_clusters']
        
        # Calculate fragmentation by examining cluster sizes
        labels = eval_data['labels']
        cluster_sizes = []
        for cluster_id in set(labels):
            cluster_size = np.sum(labels == cluster_id)
            cluster_sizes.append(cluster_size)
        
        # Calculate fragmentation metrics
        single_item_clusters = sum(1 for size in cluster_sizes if size == 1)
        fragmentation_ratio = single_item_clusters / n_clusters if n_clusters > 0 else 0
        
        # Apply penalty if fragmentation is excessive (>25% single-item clusters)
        penalty = 0
        if fragmentation_ratio > 0.25:
            # Scale penalty based on how much fragmentation exceeds threshold
            excess_fragmentation = fragmentation_ratio - 0.25
            penalty = excess_fragmentation * 0.3  # 30% penalty weight
        
        # Calculate adjusted score
        adjusted_score = original_score - penalty
        
        logger.info(f"  Fine candidate: {n_clusters} clusters, "
                   f"original_score={original_score:.3f}, "
                   f"fragmentation={fragmentation_ratio:.2%}, "
                   f"penalty={penalty:.3f}, "
                   f"adjusted_score={adjusted_score:.3f}")
        
        # Track best candidate
        if adjusted_score > best_adjusted_score:
            best_adjusted_score = adjusted_score
            best_candidate = (height, eval_data)
    
    if best_candidate:
        height, eval_data = best_candidate
        logger.info(f"Selected fine level: {eval_data['n_clusters']} clusters with adjusted score {best_adjusted_score:.3f}")
        return best_candidate
    
    return None


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

# Old artificial hierarchy consistency function removed - natural breaks are inherently consistent

# Removed select_best_candidate_for_level - no longer applying artificial penalties

# Old ensure_hierarchy_consistency function removed - using intelligent version

# Removed old DBSCAN and quality calculation functions - now using intelligent hierarchical only


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
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_dendrogram_visualization(embeddings, filenames, output_file):
    """Create and save dendrogram visualization with cluster extraction"""
    logger.info("Building hierarchical clustering for dendrogram...")
    
    # Calculate distance matrix
    distance_matrix = cosine_distances(embeddings)
    condensed_distances = squareform(distance_matrix)
    
    # Create linkage matrix using ward method
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    logger.info("Creating dendrogram plot...")
    
    # Create large figure for dendrogram
    plt.figure(figsize=(15, 8))
    
    # Create dendrogram
    dendrogram_result = dendrogram(
        linkage_matrix,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8
    )
    
    plt.title(f'Hierarchical Clustering Dendrogram\n({len(embeddings)} Voice Memo Embeddings)', 
              fontsize=14, pad=20)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cosine Distance', fontsize=12)
    
    # Add grid for easier reading
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show some statistics
    heights = linkage_matrix[:, 2]
    logger.info(f"Dendrogram statistics:")
    logger.info(f"  Min merge distance: {np.min(heights):.3f}")
    logger.info(f"  Max merge distance: {np.max(heights):.3f}")
    logger.info(f"  Mean merge distance: {np.mean(heights):.3f}")
    
    # Extract different cluster levels from dendrogram
    dendrogram_clusters = extract_dendrogram_clusters(linkage_matrix, filenames)
    
    # Save dendrogram cluster data
    cluster_data_path = output_file.parent / "dendrogram_clusters.json"
    with open(cluster_data_path, 'w', encoding='utf-8') as f:
        json.dump(dendrogram_clusters, f, indent=2)
    logger.info(f"Dendrogram cluster data saved to: {cluster_data_path}")
    
    plt.close()
    return output_file, dendrogram_clusters


def extract_dendrogram_clusters(linkage_matrix, filenames):
    """Extract cluster assignments at different dendrogram cut heights"""
    heights = linkage_matrix[:, 2]
    
    # Create different cut levels - more granular options
    cut_heights = [
        np.percentile(heights, 95),  # Very fine clusters (Level 1)
        np.percentile(heights, 85),  # Fine clusters (Level 2)
        np.percentile(heights, 70),  # Medium-fine clusters (Level 3)  
        np.percentile(heights, 55),  # Medium clusters (Level 4)
        np.percentile(heights, 40),  # Coarse clusters (Level 5)
        np.percentile(heights, 25),  # Very coarse clusters (Level 6)
    ]
    
    dendrogram_clusters = {
        "metadata": {
            "total_files": len(filenames),
            "linkage_method": "ward",
            "distance_metric": "cosine",
            "generated_at": datetime.now().isoformat()
        },
        "levels": {}
    }
    
    for i, height in enumerate(cut_heights):
        labels = fcluster(linkage_matrix, height, criterion='distance') - 1
        n_clusters = len(np.unique(labels))
        
        # Organize files by cluster
        clusters = {}
        for filename, label in zip(filenames, labels):
            # Convert numpy int to Python int for JSON serialization
            label_key = int(label)
            if label_key not in clusters:
                clusters[label_key] = []
            clusters[label_key].append(filename)
        
        level_name = f"level_{i+1}"
        dendrogram_clusters["levels"][level_name] = {
            "cut_height": float(height),
            "n_clusters": n_clusters,
            "clusters": clusters,
            "labels": [int(x) for x in labels],
            "description": f"Level {i+1}: {n_clusters} clusters"
        }
        
        logger.info(f"Dendrogram level {i+1}: {n_clusters} clusters at height {height:.3f}")
    
    return dendrogram_clusters


def create_2d_visualization(embeddings, labels, filenames, method='TSNE', output_file=None, clustering_name="default"):
    """Create 2D visualization using TSNE or UMAP with coordinate saving"""
    logger.info(f"Creating {method} 2D visualization for {clustering_name}...")
    
    # Reduce to 2D for visualization
    if method.upper() == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, metric='cosine', perplexity=min(30, len(embeddings)-1))
    elif method.upper() == 'UMAP':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    else:
        raise ValueError("Method must be 'TSNE' or 'UMAP'")
    
    coords_2d = reducer.fit_transform(embeddings)
    
    # Save coordinates data for interactive visualization
    coord_data = {
        "method": method,
        "clustering_name": clustering_name,
        "coordinates": coords_2d.tolist(),
        "labels": labels.tolist(),
        "filenames": filenames,
        "n_clusters": len(np.unique(labels[labels != -1])) if -1 in labels else len(np.unique(labels)),
        "generated_at": datetime.now().isoformat()
    }
    
    if output_file:
        # Save coordinate data
        coord_file = output_file.parent / f"{clustering_name}_{method.lower()}_coordinates.json"
        with open(coord_file, 'w', encoding='utf-8') as f:
            json.dump(coord_data, f, indent=2)
        logger.info(f"Saved {method} coordinates to: {coord_file}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 9))
    
    # Plot each cluster in different color
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    # Use a colormap that works well with many clusters
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
    
    color_idx = 0
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Plot noise points in black
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=[colors[color_idx]], s=50, alpha=0.7, label=f'Cluster {label}')
            color_idx += 1
    
    # Enhanced title with clustering info
    title = f'{clustering_name} - {method} Visualization\n{n_clusters} clusters, {len(embeddings)} total points'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    
    # Only show legend if not too many clusters
    if n_clusters <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.close()
    return output_file, coord_data


def process_visualizations():
    """Generate clustering visualizations: dendrogram, TSNE, and UMAP"""
    print("📊 Starting Clustering Visualization Process")
    print("=" * 60)
    
    # Check if clustering results exist
    clustering_results_path = Path("data/clustering_results.json")
    if not clustering_results_path.exists():
        logger.error("Clustering results not found: data/clustering_results.json")
        logger.error("Run 'cluster' command first to generate clustering results")
        return 1
    
    # Load clustering results
    print("📂 Loading clustering results...")
    with open(clustering_results_path, 'r', encoding='utf-8') as f:
        clustering_data = json.load(f)
    
    # Load embeddings
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        logger.error("Embeddings directory not found: data/embeddings")
        return 1
    
    embedding_files = list(embeddings_dir.rglob("*.json"))
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    embeddings = []
    filenames = []
    processed = 0
    
    print("Loading embeddings...")
    for embedding_path in embedding_files:
        try:
            with open(embedding_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            
            if 'embedding_vector' in embedding_data:
                embeddings.append(embedding_data['embedding_vector'])
                relative_path = embedding_path.relative_to(embeddings_dir)
                filenames.append(str(relative_path))
                processed += 1
                
        except Exception as e:
            logger.error(f"Error loading {embedding_path}: {e}")
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings loaded")
        return 1
    
    embeddings = np.array(embeddings)
    labels = np.array(clustering_data['clustering_results']['labels'])
    
    logger.info(f"Loaded {processed} embeddings for visualization")
    
    # Create output directory
    viz_dir = Path("data/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    visualizations_created = []
    
    try:
        # 1. Create dendrogram with cluster extraction
        print("\n🌳 Creating dendrogram...")
        dendrogram_path = viz_dir / "dendrogram.png"
        dendrogram_file, dendrogram_clusters = create_dendrogram_visualization(embeddings, filenames, dendrogram_path)
        visualizations_created.append(("Dendrogram", dendrogram_path))
        print(f"✅ Dendrogram saved to: {dendrogram_path}")
        
        # 2. Create t-SNE visualization for best clustering
        print("\n🎯 Creating t-SNE visualization...")
        tsne_path = viz_dir / "best_clustering_tsne.png"
        best_clustering_name = f"{clustering_data['clustering_results']['method']}_best"
        tsne_file, tsne_coords = create_2d_visualization(
            embeddings, labels, filenames, 
            method='TSNE', output_file=tsne_path, 
            clustering_name=best_clustering_name
        )
        visualizations_created.append(("t-SNE 2D Best", tsne_path))
        print(f"✅ t-SNE visualization saved to: {tsne_path}")
        
        # 3. Create UMAP visualization for best clustering (if available)
        if UMAP_AVAILABLE:
            print("\n🗺️  Creating UMAP visualization...")
            umap_path = viz_dir / "best_clustering_umap.png"
            umap_file, umap_coords = create_2d_visualization(
                embeddings, labels, filenames,
                method='UMAP', output_file=umap_path,
                clustering_name=best_clustering_name
            )
            visualizations_created.append(("UMAP 2D Best", umap_path))
            print(f"✅ UMAP visualization saved to: {umap_path}")
        else:
            print("\n⚠️  UMAP not available, skipping UMAP visualization")
            print("   Install with: pip install umap-learn")
        
        # Create summary
        summary = {
            "visualizations": [
                {
                    "type": viz_type,
                    "file_path": str(path),
                    "file_size_kb": path.stat().st_size / 1024
                }
                for viz_type, path in visualizations_created
            ],
            "clustering_method": clustering_data['clustering_results']['method'],
            "n_clusters": clustering_data['clustering_results']['n_clusters'],
            "quality_score": clustering_data['clustering_results']['quality_score'],
            "total_embeddings": len(embeddings),
            "generated_at": datetime.now().isoformat()
        }
        
        summary_path = viz_dir / "visualization_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Visualization complete!")
        print(f"📁 All visualizations saved to: {viz_dir}")
        print(f"📋 Summary saved to: {summary_path}")
        
        for viz_type, path in visualizations_created:
            file_size = path.stat().st_size / 1024
            print(f"   {viz_type}: {path} ({file_size:.1f} KB)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def process_multi_algorithm_visualizations():
    """Generate 2D visualizations for all meaningful clustering algorithms"""
    print("🔍 Starting Multi-Algorithm Visualization Process")
    print("=" * 60)
    
    # Load embeddings first
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        logger.error("Embeddings directory not found: data/embeddings")
        return 1
    
    embedding_files = list(embeddings_dir.rglob("*.json"))
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    embeddings = []
    filenames = []
    
    print("Loading embeddings...")
    for embedding_path in embedding_files:
        try:
            with open(embedding_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            
            if 'embedding_vector' in embedding_data:
                embeddings.append(embedding_data['embedding_vector'])
                relative_path = embedding_path.relative_to(embeddings_dir)
                filenames.append(str(relative_path))
                
        except Exception as e:
            logger.error(f"Error loading {embedding_path}: {e}")
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings loaded")
        return 1
    
    embeddings = np.array(embeddings)
    logger.info(f"Loaded {len(embeddings)} embeddings")
    
    # Load all clustering attempts
    clusters_dir = Path("data/clusters")
    summary_path = clusters_dir / "all_attempts_summary.json"
    
    if not summary_path.exists():
        logger.error("Clustering summary not found. Run clustering first.")
        return 1
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Filter to meaningful clustering results (30+ clusters)
    meaningful_attempts = [
        attempt for attempt in summary['attempts'] 
        if attempt['n_clusters'] >= 30
    ]
    
    print(f"Found {len(meaningful_attempts)} meaningful clustering approaches")
    
    viz_dir = Path("data/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    all_coord_data = []
    
    for i, attempt in enumerate(meaningful_attempts):
        print(f"\n📊 Processing clustering {i+1}/{len(meaningful_attempts)}: {attempt['method']} ({attempt['n_clusters']} clusters)")
        
        # Load this clustering result
        cluster_file = clusters_dir / attempt['filename']
        if not cluster_file.exists():
            logger.warning(f"Cluster file not found: {cluster_file}")
            continue
            
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        labels = np.array(cluster_data['clustering_results']['labels'])
        clustering_name = f"{attempt['method']}_n{attempt['n_clusters']}"
        
        # Create t-SNE visualization
        print(f"  🎯 Creating t-SNE for {clustering_name}...")
        tsne_path = viz_dir / f"{clustering_name}_tsne.png"
        try:
            tsne_file, tsne_coords = create_2d_visualization(
                embeddings, labels, filenames,
                method='TSNE', output_file=tsne_path,
                clustering_name=clustering_name
            )
            all_coord_data.append(tsne_coords)
        except Exception as e:
            logger.error(f"Failed to create t-SNE for {clustering_name}: {e}")
            continue
        
        # Create UMAP visualization if available
        if UMAP_AVAILABLE:
            print(f"  🗺️  Creating UMAP for {clustering_name}...")
            umap_path = viz_dir / f"{clustering_name}_umap.png"
            try:
                umap_file, umap_coords = create_2d_visualization(
                    embeddings, labels, filenames,
                    method='UMAP', output_file=umap_path,
                    clustering_name=clustering_name
                )
                all_coord_data.append(umap_coords)
            except Exception as e:
                logger.error(f"Failed to create UMAP for {clustering_name}: {e}")
    
    # Save combined coordinate data for interactive website
    combined_data = {
        "meaningful_clusterings": meaningful_attempts,
        "coordinate_files": [f"{data['clustering_name']}_{data['method'].lower()}_coordinates.json" for data in all_coord_data],
        "total_embeddings": len(embeddings),
        "generated_at": datetime.now().isoformat()
    }
    
    combined_path = viz_dir / "multi_algorithm_summary.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\n🎉 Multi-algorithm visualization complete!")
    print(f"📁 Generated {len(all_coord_data)} coordinate files")
    print(f"📊 Visualized {len(meaningful_attempts)} clustering approaches")
    print(f"📋 Summary saved to: {combined_path}")
    
    return 0


def process_clustering():
    """Process embeddings to generate clusters."""
    print("🧲 Starting Clustering Process")
    print("=" * 60)
    
    # Collect all embedding files
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        logger.error("Embeddings directory not found: data/embeddings")
        logger.error("Run 'embeddings' command first to generate embeddings")
        return 1
    
    embedding_files = list(embeddings_dir.rglob("*.json"))
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    if len(embedding_files) == 0:
        logger.error("No embedding files found")
        logger.error("Run 'embeddings' command first to generate embeddings")
        return 1
    
    # Load all embeddings
    embeddings = []
    filenames = []
    processed = 0
    failed = 0
    
    print("Loading embeddings...")
    for embedding_path in embedding_files:
        try:
            with open(embedding_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            
            # Extract embedding vector
            if 'embedding_vector' in embedding_data:
                embeddings.append(embedding_data['embedding_vector'])
                # Use relative path as identifier
                relative_path = embedding_path.relative_to(embeddings_dir)
                filenames.append(str(relative_path))
                processed += 1
            else:
                logger.warning(f"No embedding_vector found in: {embedding_path}")
                failed += 1
                
        except Exception as e:
            logger.error(f"Error loading {embedding_path}: {e}")
            failed += 1
    
    logger.info(f"Loaded {processed} embeddings, {failed} failed")
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings loaded")
        return 1
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Run intelligent hierarchical clustering
    print("\n🔬 Running intelligent hierarchical clustering...")
    print("Using 3-phase approach: natural breakpoints → quality validation → level assignment")
    
    try:
        result = intelligent_hierarchical_clustering(embeddings, filenames)
        
        # Display results
        print(f"\n✅ INTELLIGENT HIERARCHICAL CLUSTERING COMPLETE")
        print("=" * 60)
        
        # Show recommendations for each level (use dynamic level names from natural break detection)
        print(f"\n📊 SEMANTIC HIERARCHY LEVELS:")
        dynamic_levels = list(result['recommendations'].keys()) if result['recommendations'] else []
        level_mapping = {'coarse': 'Major', 'medium': 'Sub', 'fine': 'Specific'}
        
        for level in dynamic_levels:
            rec = result['recommendations'].get(level)
            if rec:
                display_name = level_mapping.get(level, level.title())
                print(f"\n🎯 {display_name} Themes:")
                print(f"   Cut Height: {rec['height']:.3f}")
                print(f"   Clusters: {rec['n_clusters']}")
                print(f"   Quality: {rec['quality_score']:.3f}")
                print(f"   Silhouette: {rec['silhouette']:.3f}")
                
                # Show cluster distribution for this level
                labels = np.array(rec['labels'])
                unique_labels = np.unique(labels)
                print(f"   Distribution:")
                for label in unique_labels[:5]:  # Show first 5 clusters
                    count = np.sum(labels == label)
                    print(f"     Cluster {label}: {count} items")
                if len(unique_labels) > 5:
                    print(f"     ... and {len(unique_labels)-5} more clusters")
            else:
                print(f"\n❌ {level.title()} Themes: No suitable height found")
        
        # Fallback to old names if no dynamic levels found
        if not dynamic_levels:
            for level in ['major', 'sub', 'specific']:
                print(f"\n❌ {level.title()} Themes: No suitable height found")
        
        # Calculate total analyzed vs. found homes
        total_analyzed = len(filenames)
        total_with_homes = 0
        total_outliers = 0
        
        # Count from the finest level (highest granularity) - use dynamic level names
        finest_level_name = None
        finest_cluster_count = 0
        
        # Find the level with most clusters (finest granularity)
        for level_name, level_data in result['recommendations'].items():
            if level_data and level_data['n_clusters'] > finest_cluster_count:
                finest_cluster_count = level_data['n_clusters']
                finest_level_name = level_name
        
        if finest_level_name:
            finest_rec = result['recommendations'][finest_level_name]
            labels = np.array(finest_rec['labels'])
            # Count items that are not outliers (assuming outliers would be labeled as -1 or in very small clusters)
            cluster_sizes = {}
            for label in labels:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
            # Items in clusters of size >= 2 are considered "found homes"
            for cluster_id, size in cluster_sizes.items():
                if size >= 2:  # Clusters with 2+ items are meaningful homes
                    total_with_homes += size
                else:
                    total_outliers += size
        else:
            total_with_homes = total_analyzed  # Fallback if no levels found
        
        # Create detailed results with all three levels
        detailed_results = {
            "clustering_type": "intelligent_hierarchical",
            "linkage_matrix": result['linkage_matrix'].tolist(),
            "file_mapping": filenames,
            "optimal_levels": {},
            "all_evaluations": {},
            "summary": {
                "total_analyzed": total_analyzed,
                "total_with_homes": total_with_homes,
                "total_outliers": total_outliers,
                "method": "hierarchical_ward",
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Convert optimal_levels to the array format expected by the website
        optimal_levels_array = []
        
        # Save each level's clustering (use dynamic level names)
        for level_name, rec in result['recommendations'].items():
            if rec:
                # Group files by cluster for this level
                clusters = {}
                labels = rec['labels']
                for i, label in enumerate(labels):
                    label_key = int(label - 1)  # Convert to 0-indexed
                    if label_key not in clusters:
                        clusters[label_key] = []
                    clusters[label_key].append(filenames[i])
                
                # Store in both formats: dictionary for backward compatibility, array for website
                detailed_results["optimal_levels"][level_name] = {
                    "height": rec['height'],
                    "n_clusters": rec['n_clusters'],
                    "quality_score": rec['quality_score'],
                    "silhouette": rec['silhouette'],
                    "labels": [int(x - 1) for x in labels],  # Convert to 0-indexed
                    "clusters": clusters,
                    "description": rec['description']
                }
                
                # Also create array format for website
                optimal_levels_array.append({
                    "level": level_name,
                    "clusters": rec['n_clusters'],
                    "quality_score": rec['quality_score'],
                    "silhouette": rec['silhouette'],
                    "height": rec['height']
                })
        
        # Store array format for website consumption
        detailed_results["optimal_levels_array"] = optimal_levels_array
        
        # Save evaluation data for all tested heights
        for height, eval_data in result['all_evaluations'].items():
            detailed_results["all_evaluations"][str(height)] = {
                "n_clusters": eval_data['n_clusters'],
                "quality_score": eval_data['metrics']['composite']['score'],
                "silhouette": eval_data['metrics']['silhouette']['average'],
                "balance_score": eval_data['metrics']['balance']['balance_score'],
                "separation_ratio": eval_data['metrics']['separation']['ratio']
            }
        
        # Save intelligent hierarchical results to clusters directory
        clusters_dir = Path("data/clusters")
        clusters_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = clusters_dir / "intelligent_hierarchical_results.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Also save a simplified version for compatibility in clusters directory
        simple_results = {
            "clustering_results": {
                "method": "intelligent_hierarchical",
                "optimal_levels": detailed_results["optimal_levels"]
            },
            "file_mapping": filenames,
            "summary": detailed_results["summary"]
        }
        
        simple_path = clusters_dir / "clustering_results.json"
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Intelligent hierarchical clustering complete!")
        print(f"📊 Analysis Summary:")
        print(f"   Total voice memos analyzed: {total_analyzed}")
        print(f"   Found meaningful homes: {total_with_homes}")
        print(f"   Outliers (single-item clusters): {total_outliers}")
        print(f"   Coverage rate: {total_with_homes/total_analyzed*100:.1f}%")
        print(f"📁 Full results saved to: {output_path}")
        print(f"📊 Compatibility results saved to: {simple_path}")
        print(f"💡 Use intelligent_hierarchical_website.py to explore the semantic hierarchy!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def process_cluster_naming():
    """Process hierarchical clusters to generate titles and descriptions using Claude."""
    print("🏷️  Starting Cluster Naming Process")
    print("=" * 60)
    
    # Check if clustering results exist
    clusters_dir = Path("data/clusters")
    results_path = clusters_dir / "intelligent_hierarchical_results.json"
    
    if not results_path.exists():
        logger.error("Hierarchical clustering results not found!")
        logger.error("Run 'cluster' command first to generate clustering results")
        return 1
    
    # Get Claude API key
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        logger.error("CLAUDE_API_KEY not found in environment!")
        return 1
    
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load clustering results
    print("📂 Loading hierarchical clustering results...")
    with open(results_path, 'r', encoding='utf-8') as f:
        clustering_data = json.load(f)
    
    optimal_levels = clustering_data.get('optimal_levels', {})
    
    if not optimal_levels:
        logger.error("No optimal levels found in clustering results")
        return 1
    
    # Collect raw essences for each cluster
    print("📖 Collecting raw essences from fingerprints...")
    fingerprints_dir = Path("data/fingerprints")
    
    cluster_essences = {}
    total_clusters = 0
    
    for level in ['coarse', 'medium', 'fine']:
        level_data = optimal_levels.get(level)
        if not level_data:
            continue
            
        clusters = level_data.get('clusters', {})
        cluster_essences[level] = {}
        
        for cluster_id, file_list in clusters.items():
            essences = []
            
            for file_path in file_list:
                # Convert .json embedding path to fingerprint path  
                clean_path = file_path.replace('.json', '.json') if file_path.endswith('.json') else file_path + '.json'
                fingerprint_path = fingerprints_dir / clean_path
                
                try:
                    with open(fingerprint_path, 'r', encoding='utf-8') as f:
                        fingerprint_data = json.load(f)
                    
                    raw_essence = fingerprint_data.get('raw_essence', '')
                    if raw_essence:
                        essences.append(raw_essence)
                        
                except Exception as e:
                    logger.warning(f"Could not load fingerprint for {file_path}: {e}")
            
            if essences:
                cluster_essences[level][cluster_id] = {
                    'member_count': len(file_list),
                    'raw_essences': essences
                }
                total_clusters += 1
                
        logger.info(f"Level {level}: {len(clusters)} clusters with essences collected")
    
    print(f"📊 Collected essences for {total_clusters} total clusters")
    
    # Build Claude prompt
    cluster_prompt = """You are an expert at analyzing patterns in voice memo insights and creating concise, meaningful titles and descriptions for thematic clusters.

## DATA CONTEXT:
This data comes from personal voice memos - spontaneous audio recordings of thoughts, ideas, reflections, and insights. Each voice memo was:
1. Transcribed from audio to text
2. Processed to extract semantic "essence" - the core meaning/insight
3. Embedded into high-dimensional vectors
4. Clustered using mathematical natural break detection

You're seeing the RAW ESSENCES (core insights) from voice memos that were mathematically determined to be semantically similar.

## HIERARCHY LEVELS:
I will provide clusters at three naturally-discovered levels:
- COARSE themes (broadest categories) 
- MEDIUM themes (medium granularity)
- FINE topics (most specific)

## YOUR TASK:
For each cluster, create:
1. **title**: The title should be semantically relevant and instantly spark recognition and resonance in the person who's reading it. They should know exactly what it means.
2. **description**: The description should spark awe and understanding, using words and language that resonates with the person who originally spoke these insights.

Return ONLY a JSON object with this structure:
```json
{
  "coarse": {
    "0": {"title": "...", "description": "..."},
    "1": {"title": "...", "description": "..."}
  },
  "medium": {
    "0": {"title": "...", "description": "..."},
    "1": {"title": "...", "description": "..."}
  },
  "fine": {
    "0": {"title": "...", "description": "..."},
    "1": {"title": "...", "description": "..."}
  }
}
```

Here are the clusters with their raw essences:

"""
    
    # Add cluster data to prompt
    for level in ['coarse', 'medium', 'fine']:
        if level not in cluster_essences:
            continue
            
        cluster_prompt += f"\n## {level.upper()} LEVEL CLUSTERS:\n\n"
        
        for cluster_id, cluster_data in cluster_essences[level].items():
            member_count = cluster_data['member_count']
            essences = cluster_data['raw_essences']
            
            # Skip clusters with only 1-2 members
            if member_count <= 2:
                continue
            
            cluster_prompt += f"**Cluster {cluster_id}** ({member_count} voice memos):\n"
            
            for i, essence in enumerate(essences[:10]):  # Limit to first 10 for space
                cluster_prompt += f"- {essence}\n"
            
            if len(essences) > 10:
                cluster_prompt += f"... and {len(essences) - 10} more similar insights\n"
            
            cluster_prompt += "\n"
    
    cluster_prompt += """

Remember: Return ONLY the JSON object with titles and descriptions. Make titles instantly recognizable and resonant, and descriptions that spark awe and understanding using the speaker's own language patterns."""

    try:
        print("🧠 Sending cluster data to Claude for analysis...")
        
        # Use streaming for large requests
        response_text = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=64000,
            temperature=0.3,
            messages=[{"role": "user", "content": cluster_prompt}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text
        
        # Parse JSON response
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                cluster_metadata = json.loads(json_str)
                
                # Validate structure
                valid_levels = ['coarse', 'medium', 'fine']
                for level in valid_levels:
                    if level not in cluster_metadata:
                        cluster_metadata[level] = {}
                
                # Save cluster metadata
                output_path = clusters_dir / "cluster_metadata.json"
                
                metadata_output = {
                    "cluster_titles_descriptions": cluster_metadata,
                    "summary": {
                        "total_clusters_named": sum(len(cluster_metadata.get(level, {})) for level in valid_levels),
                        "levels": {
                            level: len(cluster_metadata.get(level, {})) 
                            for level in valid_levels
                        },
                        "generated_at": datetime.now().isoformat(),
                        "model": "claude-3-5-sonnet-20241022"
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_output, f, indent=2, ensure_ascii=False)
                
                # Display results
                print(f"\n🎉 Cluster naming complete!")
                print(f"📊 Summary:")
                
                for level in valid_levels:
                    level_clusters = cluster_metadata.get(level, {})
                    print(f"   {level.capitalize()} level: {len(level_clusters)} clusters named")
                
                print(f"📁 Results saved to: {output_path}")
                
                # Show a few examples
                print(f"\n🏷️  Sample cluster names:")
                for level in ['coarse', 'medium']:
                    level_data = cluster_metadata.get(level, {})
                    if level_data:
                        first_cluster = list(level_data.keys())[0]
                        sample = level_data[first_cluster]
                        print(f"   {level.capitalize()} Cluster {first_cluster}: \"{sample['title']}\"")
                        print(f"      {sample['description']}")
                
                return 0
                
            else:
                raise ValueError("No JSON object found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response sample: {response_text[:500]}...")
            return 1
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return 1

def main():
    """Main execution function."""
    print("🎙️  MEGA SCRIPT: Voice Memo Processing Pipeline")
    print("=" * 60)
    
    # Simple mode selection
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nSelect mode:")
        print("1. transcribe - Transcribe audio files")
        print("2. fingerprint - Process transcripts for semantic fingerprints")
        print("3. embeddings - Generate OpenAI embeddings from fingerprints")
        print("4. cluster - Run clustering analysis on embeddings")
        print("5. name-clusters - Generate titles and descriptions for clusters using Claude")
        print("6. visualize - Generate clustering visualizations (dendrogram, t-SNE, UMAP)")
        print("7. multi-viz - Generate visualizations for all meaningful clustering algorithms")
        print("8. all - Do transcription, fingerprinting, and embeddings")
        choice = input("\nEnter choice (1/2/3/4/5/6/7/8): ").strip()
        mode = {'1': 'transcribe', '2': 'fingerprint', '3': 'embeddings', '4': 'cluster', '5': 'name-clusters', '6': 'visualize', '7': 'multi-viz', '8': 'all'}.get(choice, 'transcribe')
    
    try:
        if mode in ['transcribe', 'both', 'all']:
            # Check dependencies
            logger.info("Checking dependencies...")
            
            try:
                import whisper
                logger.info("✅ OpenAI Whisper available")
            except ImportError:
                logger.warning("⚠️  OpenAI Whisper not installed - will use Google fallback only")
                print("💡 For better results, install Whisper: pip install openai-whisper")
            
            # Check ffmpeg
            if which("ffmpeg"):
                logger.info("✅ FFmpeg available")
            else:
                logger.error("❌ FFmpeg not found - required for audio processing")
                print("Please install FFmpeg: https://ffmpeg.org/download.html")
                return 1
            
            # Start transcription
            logger.info("Starting mega transcription process...")
            summary = process_all_files(max_workers=3)  # Conservative parallelism
            
            print("\n🎉 Transcription Completed!")
            print(f"✅ Successful: {summary['stats']['successful']}")
            print(f"❌ Failed: {summary['stats']['failed']}")
            print(f"⏭️  Skipped: {summary['stats']['skipped']}")
            print(f"📁 Transcripts saved to: data/transcripts/")
        
        if mode in ['fingerprint', 'both', 'all']:
            print("\n" + "=" * 60)
            # Process semantic fingerprints
            result = process_semantic_fingerprints()
            if result != 0:
                return result
        
        if mode in ['embeddings', 'all']:
            print("\n" + "=" * 60)
            # Process embeddings
            result = process_embeddings()
            if result != 0:
                return result
        
        if mode == 'cluster':
            print("\n" + "=" * 60)
            # Process clustering
            result = process_clustering()
            if result != 0:
                return result
        
        if mode == 'name-clusters':
            print("\n" + "=" * 60)
            # Process cluster naming
            result = process_cluster_naming()
            if result != 0:
                return result
        
        if mode == 'visualize':
            print("\n" + "=" * 60)
            # Process visualizations
            result = process_visualizations()
            if result != 0:
                return result
        
        if mode == 'multi-viz':
            print("\n" + "=" * 60)
            # Process multi-algorithm visualizations
            result = process_multi_algorithm_visualizations()
            if result != 0:
                return result
        
        print("\n🎉 MEGA SCRIPT COMPLETED!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Mega script failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())