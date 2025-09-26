#!/usr/bin/env python3
"""
Proper Transcript Processor
Uses the existing proven Claude prompts and services with thread-safe parallel processing
"""

import os
import json
import time
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from pathlib import Path
from claude_service import ClaudeService
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/transcript_processing_proper.log')
    ]
)

logger = logging.getLogger(__name__)

class ProperTranscriptProcessor:
    """
    Thread-safe transcript processor using your proven Claude prompts and services
    """
    
    def __init__(self, claude_api_key: str = None, library_path: str = "data/09 library full enhanced", max_workers: int = 3):
        """
        Initialize processor
        
        Args:
            claude_api_key: Claude API key
            library_path: Path to library data
            max_workers: Maximum number of parallel threads
        """
        self.library_path = library_path
        self.claude_api_key = claude_api_key
        self.max_workers = max_workers
        
        # Thread-safe counters
        self._lock = threading.Lock()
        self.processed_count = 0
        self.total_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        
        # Output directories
        self.temp_dir = Path("temp_results")
        self.temp_dir.mkdir(exist_ok=True)
        self.final_output = "claude_processed_transcripts.json"
        
        # Load existing processed filenames
        self.processed_filenames = self.load_processed_filenames()
    
    def load_processed_filenames(self) -> set:
        """Load filenames that have already been processed"""
        processed = set()
        
        # Check temp results directory for existing batch files
        if self.temp_dir.exists():
            for result_file in self.temp_dir.glob("results_thread_*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        for item in results:
                            processed.add(item["filename"])
                    logger.info(f"📂 Loaded {len(results)} processed filenames from {result_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {result_file}: {e}")
        
        # Check final output file if it exists
        if os.path.exists(self.final_output):
            try:
                with open(self.final_output, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    for item in results:
                        processed.add(item["filename"])
                logger.info(f"📂 Loaded {len(results)} processed filenames from {self.final_output}")
            except Exception as e:
                logger.warning(f"Failed to load {self.final_output}: {e}")
        
        if processed:
            logger.info(f"✅ Found {len(processed)} already processed transcripts to skip")
        
        return processed
        
    def load_all_transcripts(self) -> List[Dict[str, Any]]:
        """Load all transcripts from library"""
        logger.info(f"📚 Loading transcripts from {self.library_path}...")
        
        all_transcripts = []
        
        if not os.path.exists(self.library_path):
            raise FileNotFoundError(f"Library path does not exist: {self.library_path}")
        
        json_files = [f for f in os.listdir(self.library_path) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} book files")
        
        for json_file in json_files:
            file_path = os.path.join(self.library_path, json_file)
            book_title = json_file.replace('.json', '')
            
            logger.info(f"Loading book: {book_title}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Navigate through the hierarchical structure
            for chapter in book_data.get("chapters", []):
                for cluster in chapter.get("clusters", []):
                    for voice_memo in cluster.get("voice_memos", []):
                        transcript = voice_memo.get("transcript", "")
                        filename = voice_memo.get("filename", "")
                        
                        if transcript and filename:
                            # Skip if already processed
                            if filename in self.processed_filenames:
                                self.skipped_count += 1
                                logger.debug(f"⏭️ Skipping already processed: {filename}")
                                continue
                                
                            transcript_data = {
                                "filename": filename,
                                "transcript": transcript,
                                "char_count": len(transcript),
                                "word_count": len(transcript.split())
                            }
                            
                            all_transcripts.append(transcript_data)
                            self.total_count += 1
        
        logger.info(f"📊 Total NEW transcripts to process: {self.total_count}")
        if self.skipped_count > 0:
            logger.info(f"⏭️ Skipped {self.skipped_count} already processed transcripts")
        
        # Sort by size for better load balancing (smaller ones first)
        all_transcripts.sort(key=lambda x: x['char_count'])
        
        return all_transcripts
    
    def process_single_transcript(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transcript using your proven Claude service"""
        try:
            # Create a new Claude service instance for this thread
            claude_service = ClaudeService(claude_api_key=self.claude_api_key)
            
            transcript = transcript_data["transcript"]
            filename = transcript_data["filename"]
            
            logger.info(f"🤖 Processing: {filename} ({transcript_data['word_count']} words)")
            
            # Use your proven process_transcript_complete method
            result = claude_service.process_transcript_complete(
                transcript=transcript,
                filename=filename,
                audio_type="Speech"  # Default to speech
            )
            
            # Extract the data in the format you requested
            processed_data = {
                "filename": filename,
                "transcript": transcript,
                "entity_extraction": result.get("claude_tags", {}).get("keywords", ""),
                "title": result.get("title", ""),
                "notion_formatted_text": result.get("formatted_transcript", transcript)
            }
            
            # Thread-safe counter update
            with self._lock:
                self.processed_count += 1
                logger.info(f"✅ Processed ({self.processed_count}/{self.total_count}): {filename}")
            
            return processed_data
            
        except Exception as e:
            with self._lock:
                self.failed_count += 1
                logger.error(f"❌ Failed ({self.failed_count}) to process {transcript_data.get('filename', 'unknown')}: {e}")
            
            return {
                "filename": transcript_data.get("filename", "unknown"),
                "transcript": transcript_data.get("transcript", ""),
                "entity_extraction": "",
                "title": f"Processing Failed: {transcript_data.get('filename', 'unknown')}",
                "notion_formatted_text": transcript_data.get("transcript", ""),
                "error": str(e)
            }
    
    def save_thread_results(self, results: List[Dict[str, Any]], thread_id: str):
        """Save results for a specific thread"""
        output_file = self.temp_dir / f"results_thread_{thread_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Thread {thread_id} saved {len(results)} results to {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save thread {thread_id} results: {e}")
    
    def process_transcript_batch(self, transcript_batch: List[Dict[str, Any]], batch_id: str) -> str:
        """Process a batch of transcripts (used by individual threads)"""
        thread_id = f"batch_{batch_id}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🚀 Thread {thread_id} starting with {len(transcript_batch)} transcripts")
        
        thread_results = []
        
        for transcript_data in transcript_batch:
            result = self.process_single_transcript(transcript_data)
            thread_results.append(result)
            
            # Small delay between transcripts to be respectful to API
            time.sleep(0.2)
        
        # Save thread results
        self.save_thread_results(thread_results, thread_id)
        
        logger.info(f"✅ Thread {thread_id} completed with {len(thread_results)} results")
        return thread_id
    
    def create_batches(self, transcripts: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Create batches for threading"""
        batches = []
        for i in range(0, len(transcripts), batch_size):
            batch = transcripts[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def consolidate_results(self) -> List[Dict[str, Any]]:
        """Consolidate all thread results into final output"""
        logger.info("🔄 Consolidating results from all threads...")
        
        all_results = []
        result_files = list(self.temp_dir.glob("results_thread_*.json"))
        
        logger.info(f"Found {len(result_files)} thread result files")
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    thread_results = json.load(f)
                all_results.extend(thread_results)
                logger.info(f"📂 Loaded {len(thread_results)} results from {result_file.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {result_file}: {e}")
        
        # Save consolidated results
        with open(self.final_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Consolidated {len(all_results)} results to {self.final_output}")
        
        # Clean up temp files
        for result_file in result_files:
            try:
                result_file.unlink()
                logger.info(f"🗑️ Cleaned up {result_file.name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {result_file}: {e}")
        
        return all_results
    
    def process_all_transcripts(self) -> List[Dict[str, Any]]:
        """Process all transcripts using thread-safe parallel processing"""
        logger.info("🚀 Starting proper transcript processing with proven prompts...")
        
        # Load all transcripts
        all_transcripts = self.load_all_transcripts()
        
        # Check if there are any transcripts to process
        if not all_transcripts:
            logger.info("🎉 No new transcripts to process! All transcripts have been processed.")
            # Still consolidate results if they exist
            final_results = self.consolidate_results()
            return final_results
        
        # Calculate batch size based on total transcripts and worker count
        # Aim for roughly equal work distribution
        batch_size = max(1, len(all_transcripts) // (self.max_workers * 3))  # 3x more batches than workers
        logger.info(f"📊 Using batch size of {batch_size} transcripts per thread")
        
        # Create batches
        batches = self.create_batches(all_transcripts, batch_size)
        logger.info(f"📦 Created {len(batches)} batches for processing")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for batch_num, batch in enumerate(batches):
                future = executor.submit(self.process_transcript_batch, batch, batch_num)
                futures.append(future)
            
            # Wait for all threads to complete
            completed_threads = []
            for future in as_completed(futures):
                try:
                    thread_id = future.result()
                    if thread_id:
                        completed_threads.append(thread_id)
                        logger.info(f"✅ Thread {thread_id} completed successfully")
                except Exception as e:
                    logger.error(f"❌ Thread failed: {e}")
        
        logger.info(f"🎉 All {len(completed_threads)} threads completed!")
        
        # Consolidate results
        final_results = self.consolidate_results()
        
        # Final statistics
        successful = len([r for r in final_results if 'error' not in r])
        failed = len(final_results) - successful
        
        logger.info(f"📊 Final Summary:")
        logger.info(f"  ✅ Successfully processed: {successful}/{self.total_count}")
        logger.info(f"  ⏭️ Skipped (already done): {self.skipped_count}")
        logger.info(f"  ❌ Failed: {failed}")
        if self.total_count > 0:
            logger.info(f"  📈 Success rate: {(successful/self.total_count*100):.1f}%")
        
        return final_results

def main():
    """Main function"""
    try:
        # Configuration
        CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
        LIBRARY_PATH = "data/09 library full enhanced"
        MAX_WORKERS = 4  # Adjust based on your API rate limits
        
        if not CLAUDE_API_KEY:
            logger.error("❌ CLAUDE_API_KEY environment variable not set")
            return
        
        logger.info("🚀 Starting Proper Transcript Processor")
        logger.info(f"📁 Library Path: {LIBRARY_PATH}")
        logger.info(f"🧵 Max Workers: {MAX_WORKERS}")
        
        # Create processor
        processor = ProperTranscriptProcessor(
            claude_api_key=CLAUDE_API_KEY,
            library_path=LIBRARY_PATH,
            max_workers=MAX_WORKERS
        )
        
        # Process all transcripts
        results = processor.process_all_transcripts()
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"💾 Results saved to {processor.final_output}")
        logger.info(f"📊 Total processed: {len(results)} transcripts")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to process transcripts: {e}")
        raise

if __name__ == "__main__":
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("temp_results").mkdir(exist_ok=True)
    
    main()