"""
Audio file processor using Whisper and Google Speech Recognition.

This processor is refactored from the existing mega_script.py audio transcription logic.
"""

import os
import sys
import logging
import time
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which

from ..core.base import BaseProcessor
from ..core.types import ProcessorResult


# Setup logging
logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor):
    """Handles audio files using existing Whisper/Google logic."""

    SUPPORTED_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.ogg', '.flac'}

    def __init__(self):
        """Initialize the audio processor with necessary components."""
        self.recognizer = sr.Recognizer()

        # Setup ffmpeg for pydub
        AudioSegment.converter = which("ffmpeg")
        AudioSegment.ffmpeg = which("ffmpeg")
        AudioSegment.ffprobe = which("ffprobe")

        # Thread-safe lock for stats (if needed in the future)
        self.lock = threading.Lock()

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def process(self, file_path: Path, output_dir: Path) -> ProcessorResult:
        """Process audio file and extract text."""
        start_time = time.time()
        output_file = output_dir / f"{file_path.stem}.txt"

        try:
            # Skip if output already exists
            if output_file.exists():
                logger.info(f"Skipping (already exists): {file_path.name}")
                return ProcessorResult(
                    success=True,
                    text="",
                    source_file=file_path,
                    output_file=output_file,
                    processor_type="audio",
                    processing_time=0,
                    error_message="File already exists"
                )

            # Get file info
            file_size = file_path.stat().st_size
            logger.info(f"Transcribing: {file_path.name} ({file_size:,} bytes)")

            # Try Whisper first (preferred)
            try:
                transcript = self.transcribe_with_whisper_local(str(file_path))
                transcription_method = "whisper_local"
            except Exception as whisper_error:
                logger.warning(f"Whisper failed for {file_path.name}: {whisper_error}")
                # Fall back to Google Speech Recognition
                try:
                    transcript = self.fallback_transcribe_google(str(file_path))
                    transcription_method = "google_fallback"
                except Exception as fallback_error:
                    logger.error(f"All transcription methods failed for {file_path.name}: {fallback_error}")
                    transcript = "[TRANSCRIPTION COMPLETELY FAILED]"
                    transcription_method = "failed"

            # Save transcript to text file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)

            duration = time.time() - start_time

            # Check if transcription actually failed
            if transcript.startswith("[") and transcript.endswith("FAILED]"):
                return ProcessorResult(
                    success=False,
                    text=transcript,
                    source_file=file_path,
                    output_file=None,
                    processor_type="audio",
                    processing_time=duration,
                    error_message=transcript
                )

            logger.info(f"✅ Completed: {file_path.name} ({duration:.1f}s, {len(transcript)} chars)")

            return ProcessorResult(
                success=True,
                text=transcript,
                source_file=file_path,
                output_file=output_file,
                processor_type="audio",
                processing_time=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error transcribing {file_path.name}: {e}"
            logger.error(error_msg)

            return ProcessorResult(
                success=False,
                text="",
                source_file=file_path,
                output_file=None,
                processor_type="audio",
                processing_time=duration,
                error_message=str(e)
            )

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