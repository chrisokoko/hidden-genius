"""
Progress Tracker for Hierarchical Notion System
Tracks creation progress and enables resume capability
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Tracks progress of database and page creation, including audio upload status.
    Enables resume capability after crashes or interruptions.
    """
    
    def __init__(self, progress_file: str = "notion_progress.json"):
        """
        Initialize progress tracker
        
        Args:
            progress_file: Path to JSON file for saving progress
        """
        self.progress_file = progress_file
        self.progress_data = self._load_progress()
        
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress from file or create new structure"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"📂 Loaded existing progress from {self.progress_file}")
                return data
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting fresh.")
                return self._create_empty_progress()
        else:
            logger.info("📝 Starting with fresh progress tracking")
            return self._create_empty_progress()
    
    def _create_empty_progress(self) -> Dict[str, Any]:
        """Create empty progress structure"""
        return {
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "database_ids": {
                "books": None,
                "chapters": None,
                "stories": None,
                "voice_memos": None
            },
            "created_pages": {
                "books": {},      # book_title -> page_id
                "chapters": {},   # chapter_key -> page_id
                "stories": {},    # story_key -> page_id
                "voice_memos": {} # filename -> page_id
            },
            "audio_uploads": {
                "successful": {},  # filename -> {"page_id": id, "uploaded_at": timestamp}
                "failed": {}       # filename -> {"page_id": id, "error": error, "attempts": count}
            },
            "statistics": {
                "total_books": 0,
                "total_chapters": 0,
                "total_stories": 0,
                "total_voice_memos": 0,
                "successful_audio_uploads": 0,
                "failed_audio_uploads": 0
            }
        }
    
    def _save_progress(self):
        """Save progress to file atomically"""
        try:
            self.progress_data["last_updated"] = datetime.now().isoformat()
            
            # Write to temp file first for atomic operation
            temp_file = f"{self.progress_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
            
            # Rename temp file to actual file (atomic on most systems)
            os.replace(temp_file, self.progress_file)
            logger.debug(f"💾 Progress saved to {self.progress_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    # Database tracking methods
    
    def save_database_ids(self, database_ids: Dict[str, str]):
        """Save database IDs after creation"""
        self.progress_data["database_ids"] = database_ids
        self._save_progress()
        logger.info(f"💾 Saved database IDs: {database_ids}")
    
    def get_database_ids(self) -> Dict[str, Optional[str]]:
        """Get saved database IDs"""
        return self.progress_data.get("database_ids", {})
    
    def has_databases(self) -> bool:
        """Check if databases have been created"""
        ids = self.get_database_ids()
        return all(ids.get(db) for db in ["books", "chapters", "stories", "voice_memos"])
    
    # Page tracking methods
    
    def save_page_created(self, page_type: str, key: str, page_id: str, metadata: Optional[Dict] = None):
        """
        Save that a page was created
        
        Args:
            page_type: Type of page (books, chapters, stories, voice_memos)
            key: Unique key for the page (e.g., filename for voice memos)
            page_id: Notion page ID
            metadata: Optional metadata about the page
        """
        if page_type not in self.progress_data["created_pages"]:
            self.progress_data["created_pages"][page_type] = {}
        
        self.progress_data["created_pages"][page_type][key] = {
            "page_id": page_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update statistics
        stat_key = f"total_{page_type}"
        if stat_key in self.progress_data["statistics"]:
            self.progress_data["statistics"][stat_key] = len(self.progress_data["created_pages"][page_type])
        
        self._save_progress()
        logger.debug(f"💾 Saved {page_type} page: {key} -> {page_id}")
    
    def is_page_created(self, page_type: str, key: str) -> bool:
        """Check if a page has already been created"""
        return key in self.progress_data["created_pages"].get(page_type, {})
    
    def get_page_id(self, page_type: str, key: str) -> Optional[str]:
        """Get the page ID for a previously created page"""
        page_data = self.progress_data["created_pages"].get(page_type, {}).get(key, {})
        if isinstance(page_data, dict):
            return page_data.get("page_id")
        return page_data  # For backward compatibility if storing just ID
    
    # Audio upload tracking methods
    
    def save_audio_upload_success(self, filename: str, page_id: str):
        """Save successful audio upload"""
        self.progress_data["audio_uploads"]["successful"][filename] = {
            "page_id": page_id,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Remove from failed if it was there
        if filename in self.progress_data["audio_uploads"]["failed"]:
            del self.progress_data["audio_uploads"]["failed"][filename]
        
        # Update statistics
        self.progress_data["statistics"]["successful_audio_uploads"] = len(
            self.progress_data["audio_uploads"]["successful"]
        )
        self.progress_data["statistics"]["failed_audio_uploads"] = len(
            self.progress_data["audio_uploads"]["failed"]
        )
        
        self._save_progress()
        logger.info(f"✅ Saved successful audio upload: {filename}")
    
    def save_audio_upload_failure(self, filename: str, page_id: str, error: str):
        """Save failed audio upload for retry"""
        if filename not in self.progress_data["audio_uploads"]["failed"]:
            self.progress_data["audio_uploads"]["failed"][filename] = {
                "page_id": page_id,
                "attempts": 0,
                "errors": []
            }
        
        failed_data = self.progress_data["audio_uploads"]["failed"][filename]
        failed_data["attempts"] += 1
        failed_data["last_error"] = error
        failed_data["last_attempt"] = datetime.now().isoformat()
        failed_data["errors"].append({
            "attempt": failed_data["attempts"],
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update statistics
        self.progress_data["statistics"]["failed_audio_uploads"] = len(
            self.progress_data["audio_uploads"]["failed"]
        )
        
        self._save_progress()
        logger.warning(f"⚠️ Saved failed audio upload: {filename} (attempt {failed_data['attempts']})")
    
    def is_audio_uploaded(self, filename: str) -> bool:
        """Check if audio file has been successfully uploaded"""
        return filename in self.progress_data["audio_uploads"]["successful"]
    
    def get_failed_uploads(self) -> Dict[str, Dict]:
        """Get all failed audio uploads for retry"""
        return self.progress_data["audio_uploads"]["failed"].copy()
    
    def get_failed_upload_count(self) -> int:
        """Get count of failed uploads"""
        return len(self.progress_data["audio_uploads"]["failed"])
    
    # Statistics and reporting
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        stats = self.progress_data["statistics"].copy()
        
        # Add completion percentages
        total_pages = sum([
            stats.get("total_books", 0),
            stats.get("total_chapters", 0),
            stats.get("total_stories", 0),
            stats.get("total_voice_memos", 0)
        ])
        
        stats["total_pages_created"] = total_pages
        
        # Audio upload success rate
        total_audio = stats.get("successful_audio_uploads", 0) + stats.get("failed_audio_uploads", 0)
        if total_audio > 0:
            stats["audio_upload_success_rate"] = (
                stats.get("successful_audio_uploads", 0) / total_audio * 100
            )
        else:
            stats["audio_upload_success_rate"] = 0
        
        return stats
    
    def print_summary(self):
        """Print a summary of current progress"""
        stats = self.get_statistics()
        
        logger.info("=" * 60)
        logger.info("📊 PROGRESS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📚 Books created: {stats.get('total_books', 0)}")
        logger.info(f"📖 Chapters created: {stats.get('total_chapters', 0)}")
        logger.info(f"📝 Stories created: {stats.get('total_stories', 0)}")
        logger.info(f"🎙️ Voice Memos created: {stats.get('total_voice_memos', 0)}")
        logger.info(f"✅ Successful audio uploads: {stats.get('successful_audio_uploads', 0)}")
        logger.info(f"❌ Failed audio uploads: {stats.get('failed_audio_uploads', 0)}")
        
        if stats.get('audio_upload_success_rate', 0) > 0:
            logger.info(f"📈 Audio upload success rate: {stats['audio_upload_success_rate']:.1f}%")
        
        logger.info("=" * 60)
    
    def reset_progress(self):
        """Reset all progress (use with caution!)"""
        logger.warning("⚠️ Resetting all progress!")
        self.progress_data = self._create_empty_progress()
        self._save_progress()
    
    def has_existing_progress(self) -> bool:
        """Check if there's existing progress to resume from"""
        return (
            self.has_databases() or
            any(self.progress_data["created_pages"].values()) or
            any(self.progress_data["audio_uploads"]["successful"])
        )