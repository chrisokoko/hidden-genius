"""
Hierarchical Notion Database System
Creates and manages four interconnected databases: Books, Chapters, Stories, and Voice Memos
"""

import os
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from notion_service import NotionService
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class HierarchicalNotionSystem:
    """
    Manages four interconnected Notion databases that mirror the JSON library structure:
    - Books Database (top level)
    - Chapters Database (linked to Books)
    - Stories Database (linked to Chapters, renamed from Clusters)
    - Voice Memos Database (linked to Stories)
    """
    
    def __init__(self, notion_token: str, parent_page_id: str, 
                 claude_data_path: str = "data/0.5 Full Transcripts/claude_processed_transcripts.json",
                 progress_tracker: Optional[ProgressTracker] = None,
                 max_workers: int = 3):
        """
        Initialize the hierarchical system
        
        Args:
            notion_token: Notion API token
            parent_page_id: ID of the parent page where databases will be created
            claude_data_path: Path to Claude processed transcripts
            progress_tracker: Optional progress tracker for resume capability
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.notion_token = notion_token
        self.parent_page_id = parent_page_id
        self.claude_data_path = claude_data_path
        
        # Initialize or use provided progress tracker
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.max_workers = max_workers
        
        # Thread-safe counters for multi-threaded operations
        self._thread_lock = threading.Lock()
        self._processed_count = 0
        self._failed_count = 0
        
        # Database IDs will be populated after creation or loaded from progress
        self.books_db_id = None
        self.chapters_db_id = None
        self.stories_db_id = None
        self.voice_memos_db_id = None
        
        # Load Claude processed data
        self.claude_data = self._load_claude_data()
        
        # Notion services for each database
        self.books_service = None
        self.chapters_service = None
        self.stories_service = None
        self.voice_memos_service = None
    
    def _load_claude_data(self) -> Dict[str, Dict[str, Any]]:
        """Load Claude processed transcripts and create filename lookup"""
        claude_lookup = {}
        
        try:
            if os.path.exists(self.claude_data_path):
                with open(self.claude_data_path, 'r', encoding='utf-8') as f:
                    claude_transcripts = json.load(f)
                
                # Create lookup by filename
                for item in claude_transcripts:
                    filename = item.get("filename", "")
                    if filename:
                        claude_lookup[filename] = {
                            "title": item.get("title", ""),
                            "entity_extraction": item.get("entity_extraction", ""),
                            "notion_formatted_text": item.get("notion_formatted_text", ""),
                            "transcript": item.get("transcript", "")
                        }
                
                logger.info(f"✅ Loaded {len(claude_lookup)} Claude processed transcripts")
            else:
                logger.warning(f"Claude data file not found: {self.claude_data_path}")
                
        except Exception as e:
            logger.error(f"Failed to load Claude data: {e}")
            
        return claude_lookup
    
    def get_claude_data_for_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get Claude processed data for a specific filename"""
        return self.claude_data.get(filename)
    
    def _upload_audio_with_retry(self, page_id: str, audio_file_path: str, filename: str, max_retries: int = 3) -> bool:
        """
        Upload audio file with retry logic
        
        Args:
            page_id: Notion page ID
            audio_file_path: Path to audio file
            filename: Original filename for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if upload successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"📤 Upload attempt {attempt + 1}/{max_retries} for {filename}")
                
                # Try to upload the audio file
                success = self.voice_memos_service.add_audio_file_to_page(page_id, audio_file_path)
                
                if success:
                    return True
                
                # If failed and not the last attempt, wait with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(f"⏳ Upload failed, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"❌ Upload attempt {attempt + 1} failed with error: {e}")
                
                # Save the specific error
                self.progress_tracker.save_audio_upload_failure(
                    filename, page_id, f"Attempt {attempt + 1}: {str(e)}"
                )
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        return False
    
    def get_audio_file_path(self, json_filename: str) -> Optional[str]:
        """
        Convert JSON filename to actual audio file path
        
        Args:
            json_filename: Filename like "03_5min_to_20min/5724 NE 60th Ave 54.json"
            
        Returns:
            Full path to audio file or None if not found
        """
        try:
            # Convert JSON filename to audio filename
            # "03_5min_to_20min/5724 NE 60th Ave 54.json" -> "03_5min_to_20min/5724 NE 60th Ave 54.m4a"
            if json_filename.endswith('.json'):
                audio_filename = json_filename[:-5] + '.m4a'  # Remove .json, add .m4a
            else:
                audio_filename = json_filename + '.m4a'  # Just add .m4a
            
            # Construct full path
            audio_file_path = os.path.join("audio_files", audio_filename)
            
            # Check if file exists
            if os.path.exists(audio_file_path):
                return os.path.abspath(audio_file_path)
            else:
                logger.debug(f"Audio file not found: {audio_file_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Error resolving audio file path for {json_filename}: {e}")
            return None
    
    def get_books_database_schema(self) -> Dict[str, Any]:
        """Define the Books database schema"""
        return {
            "Title": {
                "title": {}
            },
            "Description": {
                "rich_text": {}
            },
            "Total Chapters": {
                "number": {
                    "format": "number"
                }
            },
            "Total Stories": {
                "number": {
                    "format": "number"
                }
            },
            "Total Voice Memos": {
                "number": {
                    "format": "number"
                }
            },
            "Average Quality Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Quality Grade": {
                "select": {
                    "options": [
                        {"name": "A+", "color": "green"},
                        {"name": "A", "color": "green"},
                        {"name": "A-", "color": "green"},
                        {"name": "B+", "color": "yellow"},
                        {"name": "B", "color": "yellow"},
                        {"name": "B-", "color": "yellow"},
                        {"name": "C+", "color": "orange"},
                        {"name": "C", "color": "orange"},
                        {"name": "C-", "color": "orange"},
                        {"name": "D", "color": "red"},
                        {"name": "F", "color": "red"}
                    ]
                }
            },
            "Total Reading Time": {
                "rich_text": {}
            },
            "Total Audio Length": {
                "rich_text": {}
            },
            "Excellence Count": {
                "number": {
                    "format": "number"
                }
            },
            "Excellence Percentage": {
                "number": {
                    "format": "percent"
                }
            },
            "High Quality Count": {
                "number": {
                    "format": "number"
                }
            },
            "High Quality Percentage": {
                "number": {
                    "format": "percent"
                }
            },
            "Pillar Stage": {
                "select": {
                    "options": [
                        {"name": "BUILDING_MOMENTUM", "color": "yellow"},
                        {"name": "BREAKTHROUGH", "color": "green"},
                        {"name": "EMERGING", "color": "orange"},
                        {"name": "ESTABLISHED", "color": "blue"}
                    ]
                }
            },
            "Stage Description": {
                "rich_text": {}
            },
            "Recommended Action": {
                "rich_text": {}
            },
            "First Entry Date": {
                "date": {}
            },
            "Latest Entry Date": {
                "date": {}
            },
            "Time Span Days": {
                "number": {
                    "format": "number"
                }
            },
            "Time Span Pretty": {
                "rich_text": {}
            },
            "Created": {
                "created_time": {}
            }
        }
    
    def get_chapters_database_schema(self) -> Dict[str, Any]:
        """Define the Chapters database schema"""
        return {
            "Chapter Title": {
                "title": {}
            },
            "Part Title": {
                "rich_text": {}
            },
            "Book": {
                "relation": {
                    "database_id": self.books_db_id,
                    "single_property": {}
                }
            },
            "Story Count": {
                "number": {
                    "format": "number"
                }
            },
            "Voice Memo Count": {
                "number": {
                    "format": "number"
                }
            },
            "Average Quality Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Quality Grade": {
                "select": {
                    "options": [
                        {"name": "A+", "color": "green"},
                        {"name": "A", "color": "green"},
                        {"name": "A-", "color": "green"},
                        {"name": "B+", "color": "yellow"},
                        {"name": "B", "color": "yellow"},
                        {"name": "B-", "color": "yellow"},
                        {"name": "C+", "color": "orange"},
                        {"name": "C", "color": "orange"},
                        {"name": "C-", "color": "orange"},
                        {"name": "D", "color": "red"},
                        {"name": "F", "color": "red"}
                    ]
                }
            },
            "Total Reading Time": {
                "rich_text": {}
            },
            "Total Audio Length": {
                "rich_text": {}
            },
            "First Entry Date": {
                "date": {}
            },
            "Latest Entry Date": {
                "date": {}
            },
            "Time Span Days": {
                "number": {
                    "format": "number"
                }
            },
            "Time Span Pretty": {
                "rich_text": {}
            },
            "Created": {
                "created_time": {}
            }
        }
    
    def get_stories_database_schema(self) -> Dict[str, Any]:
        """Define the Stories database schema (renamed from Clusters)"""
        return {
            "Story Name": {
                "title": {}
            },
            "Story Description": {
                "rich_text": {}
            },
            "Chapter": {
                "relation": {
                    "database_id": self.chapters_db_id,
                    "single_property": {}
                }
            },
            "Book": {
                "rollup": {
                    "relation_property_name": "Chapter",
                    "rollup_property_name": "Book",
                    "function": "show_original"
                }
            },
            "Voice Memo Count": {
                "number": {
                    "format": "number"
                }
            },
            "Average Quality Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Quality Grade": {
                "select": {
                    "options": [
                        {"name": "A+", "color": "green"},
                        {"name": "A", "color": "green"},
                        {"name": "A-", "color": "green"},
                        {"name": "B+", "color": "yellow"},
                        {"name": "B", "color": "yellow"},
                        {"name": "B-", "color": "yellow"},
                        {"name": "C+", "color": "orange"},
                        {"name": "C", "color": "orange"},
                        {"name": "C-", "color": "orange"},
                        {"name": "D", "color": "red"},
                        {"name": "F", "color": "red"}
                    ]
                }
            },
            "Total Reading Time": {
                "rich_text": {}
            },
            "Total Audio Length": {
                "rich_text": {}
            },
            "First Entry Date": {
                "date": {}
            },
            "Latest Entry Date": {
                "date": {}
            },
            "Time Span Days": {
                "number": {
                    "format": "number"
                }
            },
            "Time Span Pretty": {
                "rich_text": {}
            },
            "Created": {
                "created_time": {}
            }
        }
    
    def get_voice_memos_database_schema(self) -> Dict[str, Any]:
        """Define the Voice Memos database schema"""
        return {
            "Title": {
                "title": {}
            },
            "Story": {
                "relation": {
                    "database_id": self.stories_db_id,
                    "single_property": {}
                }
            },
            "Filename": {
                "rich_text": {}
            },
            "Summary": {
                "rich_text": {}
            },
            "Tags": {
                "rich_text": {}
            },
            "Duration": {
                "rich_text": {}
            },
            "Duration (Seconds)": {
                "number": {}
            },
            "Audio Length Pretty": {
                "rich_text": {}
            },
            "Reading Time Pretty": {
                "rich_text": {}
            },
            "Word Count": {
                "number": {
                    "format": "number"
                }
            },
            "File Size Bytes": {
                "number": {
                    "format": "number"
                }
            },
            "Creation Date": {
                "date": {}
            },
            "Quality Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Quality Grade": {
                "select": {
                    "options": [
                        {"name": "A+", "color": "green"},
                        {"name": "A", "color": "green"},
                        {"name": "A-", "color": "green"},
                        {"name": "B+", "color": "yellow"},
                        {"name": "B", "color": "yellow"},
                        {"name": "B-", "color": "yellow"},
                        {"name": "C+", "color": "orange"},
                        {"name": "C", "color": "orange"},
                        {"name": "C-", "color": "orange"},
                        {"name": "D", "color": "red"},
                        {"name": "F", "color": "red"}
                    ]
                }
            },
            # Semantic Fingerprint Fields
            "Central Question": {
                "rich_text": {}
            },
            "Key Tension": {
                "rich_text": {}
            },
            "Breakthrough Moment": {
                "rich_text": {}
            },
            "Edge of Understanding": {
                "rich_text": {}
            },
            "Thinking Styles": {
                "multi_select": {
                    "options": [
                        {"name": "analytical-linear", "color": "blue"},
                        {"name": "systems-holistic", "color": "green"},
                        {"name": "intuitive-emergent", "color": "purple"},
                        {"name": "dialectical-synthetic", "color": "orange"},
                        {"name": "creative-associative", "color": "pink"},
                        {"name": "experiential-embodied", "color": "brown"}
                    ]
                }
            },
            "Insight Type": {
                "select": {
                    "options": [
                        {"name": "framework", "color": "blue"},
                        {"name": "methodology", "color": "green"},
                        {"name": "philosophy", "color": "purple"},
                        {"name": "observation", "color": "orange"},
                        {"name": "pattern", "color": "pink"},
                        {"name": "breakthrough", "color": "red"}
                    ]
                }
            },
            "Development Stage": {
                "select": {
                    "options": [
                        {"name": "noticing", "color": "gray"},
                        {"name": "developing", "color": "orange"},
                        {"name": "refining", "color": "yellow"},
                        {"name": "integrating", "color": "green"},
                        {"name": "breakthrough", "color": "red"},
                        {"name": "mastering", "color": "blue"}
                    ]
                }
            },
            "Connected Domains": {
                "multi_select": {
                    "options": [
                        {"name": "business_strategy", "color": "blue"},
                        {"name": "spirituality", "color": "purple"},
                        {"name": "social_change", "color": "green"},
                        {"name": "innovation", "color": "orange"},
                        {"name": "community", "color": "pink"},
                        {"name": "leadership", "color": "brown"},
                        {"name": "psychology", "color": "red"},
                        {"name": "technology", "color": "gray"}
                    ]
                }
            },
            "Uniqueness Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Depth Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Generative Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Usefulness Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Confidence Score": {
                "number": {
                    "format": "number_with_commas"
                }
            },
            "Raw Essence": {
                "rich_text": {}
            },
            "Conceptual DNA": {
                "rich_text": {}
            },
            "Audio File": {
                "files": {}
            },
            "Audio Content Type": {
                "select": {
                    "options": [
                        {"name": "Business", "color": "blue"},
                        {"name": "Personal", "color": "green"},
                        {"name": "Creative", "color": "purple"},
                        {"name": "Spiritual", "color": "pink"},
                        {"name": "Technical", "color": "gray"},
                        {"name": "Unknown", "color": "default"}
                    ]
                }
            },
            "Flagged for Deletion": {
                "checkbox": {}
            },
            "Created": {
                "created_time": {}
            }
        }
    
    def create_all_databases(self) -> Dict[str, str]:
        """
        Create all four databases in the correct order (Books -> Chapters -> Stories -> Voice Memos)
        Or load existing database IDs from progress tracker
        
        Returns:
            Dictionary mapping database names to their IDs
        """
        try:
            # Check if databases already exist in progress
            existing_ids = self.progress_tracker.get_database_ids()
            if self.progress_tracker.has_databases():
                logger.info("📂 Loading existing database IDs from progress...")
                self.books_db_id = existing_ids["books"]
                self.chapters_db_id = existing_ids["chapters"]
                self.stories_db_id = existing_ids["stories"]
                self.voice_memos_db_id = existing_ids["voice_memos"]
                
                # Initialize services with existing databases
                self.books_service = NotionService(self.books_db_id, self.notion_token)
                self.chapters_service = NotionService(self.chapters_db_id, self.notion_token)
                self.stories_service = NotionService(self.stories_db_id, self.notion_token)
                self.voice_memos_service = NotionService(self.voice_memos_db_id, self.notion_token)
                
                logger.info("✅ Using existing databases:")
                logger.info(f"  📚 Books: {self.books_db_id}")
                logger.info(f"  📖 Chapters: {self.chapters_db_id}")
                logger.info(f"  📝 Stories: {self.stories_db_id}")
                logger.info(f"  🎙️ Voice Memos: {self.voice_memos_db_id}")
                
                return {
                    "books": self.books_db_id,
                    "chapters": self.chapters_db_id,
                    "stories": self.stories_db_id,
                    "voice_memos": self.voice_memos_db_id
                }
            
            logger.info("Creating new hierarchical database system...")
            
            # Step 1: Create Books database
            logger.info("Creating Books database...")
            temp_service = NotionService(database_id="temp", notion_token=self.notion_token)  # Temporary service for creation
            books_schema = self.get_books_database_schema()
            self.books_db_id = temp_service.create_database(
                parent_id=self.parent_page_id,
                title="📚 Books",
                properties=books_schema
            )
            if not self.books_db_id:
                raise Exception("Failed to create Books database")
            logger.info(f"Books database created: {self.books_db_id}")
            
            # Step 2: Create Chapters database (references Books)
            logger.info("Creating Chapters database...")
            chapters_schema = self.get_chapters_database_schema()
            self.chapters_db_id = temp_service.create_database(
                parent_id=self.parent_page_id,
                title="📖 Chapters", 
                properties=chapters_schema
            )
            if not self.chapters_db_id:
                raise Exception("Failed to create Chapters database")
            logger.info(f"Chapters database created: {self.chapters_db_id}")
            
            # Step 3: Create Stories database (references Chapters, rollup to Books)
            logger.info("Creating Stories database...")
            stories_schema = self.get_stories_database_schema()
            self.stories_db_id = temp_service.create_database(
                parent_id=self.parent_page_id,
                title="📝 Stories",
                properties=stories_schema
            )
            if not self.stories_db_id:
                raise Exception("Failed to create Stories database")
            logger.info(f"Stories database created: {self.stories_db_id}")
            
            # Step 4: Create Voice Memos database (references Stories, rollup to Chapters and Books)
            logger.info("Creating Voice Memos database...")
            voice_memos_schema = self.get_voice_memos_database_schema()
            self.voice_memos_db_id = temp_service.create_database(
                parent_id=self.parent_page_id,
                title="🎙️ Voice Memos",
                properties=voice_memos_schema
            )
            if not self.voice_memos_db_id:
                raise Exception("Failed to create Voice Memos database")
            logger.info(f"Voice Memos database created: {self.voice_memos_db_id}")
            
            # Step 5: Initialize NotionServices for each database
            self.books_service = NotionService(self.books_db_id, notion_token=self.notion_token)
            self.chapters_service = NotionService(self.chapters_db_id, notion_token=self.notion_token)
            self.stories_service = NotionService(self.stories_db_id, notion_token=self.notion_token)
            self.voice_memos_service = NotionService(self.voice_memos_db_id, notion_token=self.notion_token)
            
            # Save database IDs to progress tracker
            database_ids = {
                "books": self.books_db_id,
                "chapters": self.chapters_db_id,
                "stories": self.stories_db_id,
                "voice_memos": self.voice_memos_db_id
            }
            self.progress_tracker.save_database_ids(database_ids)
            
            logger.info("✅ All databases created successfully!")
            
            return database_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to create databases: {e}")
            raise
    
    def load_library_data(self, library_path: str) -> Dict[str, Any]:
        """
        Load and parse library data from JSON files
        
        Args:
            library_path: Path to the library directory (e.g., "data/09 library full enhanced")
            
        Returns:
            Dictionary of book data keyed by book title
        """
        try:
            logger.info(f"Loading library data from {library_path}...")
            
            library_data = {}
            
            # Get all JSON files in the library directory
            if not os.path.exists(library_path):
                raise FileNotFoundError(f"Library path does not exist: {library_path}")
            
            json_files = [f for f in os.listdir(library_path) if f.endswith('.json')]
            logger.info(f"Found {len(json_files)} book files")
            
            for json_file in json_files:
                file_path = os.path.join(library_path, json_file)
                book_title = json_file.replace('.json', '')
                
                logger.info(f"Loading book: {book_title}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    book_data = json.load(f)
                    library_data[book_title] = book_data
            
            logger.info(f"✅ Loaded {len(library_data)} books successfully")
            return library_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load library data: {e}")
            raise
    
    def populate_databases(self, library_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Populate all databases with data from the library
        
        Args:
            library_data: Dictionary of book data
            
        Returns:
            Dictionary mapping database names to lists of created page IDs
        """
        try:
            logger.info("🚀 Starting database population...")
            
            created_pages = {
                "books": [],
                "chapters": [],
                "stories": [],
                "voice_memos": []
            }
            
            for book_title, book_data in library_data.items():
                logger.info(f"📚 Processing book: {book_title}")
                
                # Create book page
                book_page_id = self._create_book_page(book_title, book_data)
                if book_page_id:
                    created_pages["books"].append(book_page_id)
                    logger.info(f"✅ Created book page: {book_page_id}")
                    
                    # Process chapters
                    for chapter_data in book_data.get("chapters", []):
                        chapter_page_id = self._create_chapter_page(chapter_data, book_page_id)
                        if chapter_page_id:
                            created_pages["chapters"].append(chapter_page_id)
                            logger.info(f"📖 Created chapter page: {chapter_page_id}")
                            
                            # Process stories/clusters
                            for cluster_data in chapter_data.get("clusters", []):
                                story_page_id = self._create_story_page(cluster_data, chapter_page_id)
                                if story_page_id:
                                    created_pages["stories"].append(story_page_id)
                                    logger.info(f"📝 Created story page: {story_page_id}")
                                    
                                    # Process voice memos in parallel
                                    voice_memos = cluster_data.get("voice_memos", [])
                                    if voice_memos:
                                        memo_page_ids = self._process_voice_memos_parallel(voice_memos, story_page_id, max_workers=self.max_workers)
                                        created_pages["voice_memos"].extend(memo_page_ids)
                                        logger.info(f"🎙️ Created {len(memo_page_ids)} voice memo pages for story")
            
            # Summary
            logger.info("🎉 Database population completed!")
            logger.info(f"📊 Summary:")
            logger.info(f"  📚 Books: {len(created_pages['books'])}")
            logger.info(f"  📖 Chapters: {len(created_pages['chapters'])}")
            logger.info(f"  📝 Stories: {len(created_pages['stories'])}")
            logger.info(f"  🎙️ Voice Memos: {len(created_pages['voice_memos'])}")
            
            return created_pages
            
        except Exception as e:
            logger.error(f"❌ Failed to populate databases: {e}")
            raise
    
    def _create_book_page(self, book_title: str, book_data: Dict[str, Any]) -> Optional[str]:
        """Create a book page in the Books database"""
        try:
            # Check if book already exists
            if self.progress_tracker.is_page_created("books", book_title):
                page_id = self.progress_tracker.get_page_id("books", book_title)
                logger.info(f"⏭️ Skipping existing book: {book_title} ({page_id})")
                return page_id
            book_metadata = book_data.get("book_metadata", {})
            
            properties = {
                "Title": {
                    "title": [{"text": {"content": book_title}}]
                },
                "Description": {
                    "rich_text": [{"text": {"content": book_metadata.get("description", "")[:2000]}}]
                },
                "Total Chapters": {
                    "number": book_metadata.get("total_chapters", 0)
                },
                "Total Stories": {
                    "number": book_metadata.get("total_clusters", 0)  # clusters -> stories
                },
                "Total Voice Memos": {
                    "number": book_metadata.get("total_voice_memos", 0)
                },
                "Average Quality Score": {
                    "number": book_metadata.get("average_quality_coefficient", 0.0)
                },
                "Total Reading Time": {
                    "rich_text": [{"text": {"content": book_metadata.get("total_reading_time_pretty", "")}}]
                },
                "Total Audio Length": {
                    "rich_text": [{"text": {"content": book_metadata.get("total_audio_length_pretty", "")}}]
                },
                "Excellence Count": {
                    "number": book_metadata.get("excellence_count", 0)
                },
                "Excellence Percentage": {
                    "number": book_metadata.get("excellence_percentage", 0.0) / 100.0  # Convert to decimal
                },
                "High Quality Count": {
                    "number": book_metadata.get("high_quality_count", 0)
                },
                "High Quality Percentage": {
                    "number": book_metadata.get("high_quality_percentage", 0.0) / 100.0  # Convert to decimal
                },
                "Stage Description": {
                    "rich_text": [{"text": {"content": book_metadata.get("stage_description", "")[:2000]}}]
                },
                "Recommended Action": {
                    "rich_text": [{"text": {"content": book_metadata.get("recommended_action", "")[:2000]}}]
                },
                "Time Span Days": {
                    "number": book_metadata.get("time_span_days", 0)
                },
                "Time Span Pretty": {
                    "rich_text": [{"text": {"content": book_metadata.get("time_span_pretty", "")}}]
                }
            }
            
            # Add optional fields
            if book_metadata.get("pillar_stage"):
                properties["Pillar Stage"] = {"select": {"name": book_metadata["pillar_stage"]}}
            
            if book_metadata.get("first_entry_date"):
                properties["First Entry Date"] = {"date": {"start": book_metadata["first_entry_date"]}}
                
            if book_metadata.get("latest_entry_date"):
                properties["Latest Entry Date"] = {"date": {"start": book_metadata["latest_entry_date"]}}
            
            # Create the page
            page_id = self.books_service._make_api_call(
                "create_page",
                parent={"database_id": self.books_db_id},
                properties=properties,
                use_cache=False
            )["id"]
            
            # Save to progress tracker
            self.progress_tracker.save_page_created("books", book_title, page_id, book_metadata)
            
            return page_id
            
        except Exception as e:
            logger.error(f"Failed to create book page for {book_title}: {e}")
            return None
    
    def _create_chapter_page(self, chapter_data: Dict[str, Any], book_page_id: str) -> Optional[str]:
        """Create a chapter page in the Chapters database"""
        try:
            chapter_title = chapter_data.get("chapter_title", "Untitled Chapter")
            part_title = chapter_data.get("part_title", "")
            
            # Create a unique key for this chapter
            chapter_key = f"{part_title}/{chapter_title}" if part_title else chapter_title
            
            # Check if chapter already exists
            if self.progress_tracker.is_page_created("chapters", chapter_key):
                page_id = self.progress_tracker.get_page_id("chapters", chapter_key)
                logger.info(f"⏭️ Skipping existing chapter: {chapter_key} ({page_id})")
                return page_id
            
            chapter_metadata = chapter_data.get("chapter_metadata", {})
            
            properties = {
                "Chapter Title": {
                    "title": [{"text": {"content": chapter_data.get("chapter_title", "Untitled Chapter")}}]
                },
                "Part Title": {
                    "rich_text": [{"text": {"content": chapter_data.get("part_title", "")}}]
                },
                "Book": {
                    "relation": [{"id": book_page_id}]
                },
                "Story Count": {
                    "number": chapter_metadata.get("cluster_count", 0)
                },
                "Voice Memo Count": {
                    "number": chapter_metadata.get("voice_memo_count", 0)
                },
                "Average Quality Score": {
                    "number": chapter_metadata.get("average_quality_coefficient", 0.0)
                },
                "Total Reading Time": {
                    "rich_text": [{"text": {"content": chapter_metadata.get("total_reading_time_pretty", "")}}]
                },
                "Total Audio Length": {
                    "rich_text": [{"text": {"content": chapter_metadata.get("total_audio_length_pretty", "")}}]
                },
                "Time Span Days": {
                    "number": chapter_metadata.get("time_span_days", 0)
                },
                "Time Span Pretty": {
                    "rich_text": [{"text": {"content": chapter_metadata.get("time_span_pretty", "")}}]
                }
            }
            
            # Add optional fields
            if chapter_metadata.get("quality_grade_curved"):
                properties["Quality Grade"] = {"select": {"name": chapter_metadata["quality_grade_curved"]}}
                
            if chapter_metadata.get("first_entry_date"):
                properties["First Entry Date"] = {"date": {"start": chapter_metadata["first_entry_date"]}}
                
            if chapter_metadata.get("latest_entry_date"):
                properties["Latest Entry Date"] = {"date": {"start": chapter_metadata["latest_entry_date"]}}
            
            # Create the page
            page_id = self.chapters_service._make_api_call(
                "create_page",
                parent={"database_id": self.chapters_db_id},
                properties=properties,
                use_cache=False
            )["id"]
            
            # Save to progress tracker
            self.progress_tracker.save_page_created("chapters", chapter_key, page_id, chapter_metadata)
            
            return page_id
            
        except Exception as e:
            logger.error(f"Failed to create chapter page: {e}")
            return None
    
    def _create_story_page(self, cluster_data: Dict[str, Any], chapter_page_id: str) -> Optional[str]:
        """Create a story page in the Stories database"""
        try:
            story_name = cluster_data.get("cluster_name", "Untitled Story")
            
            # Create a unique key for this story (include chapter for uniqueness)
            story_key = f"{chapter_page_id}/{story_name}"
            
            # Check if story already exists
            if self.progress_tracker.is_page_created("stories", story_key):
                page_id = self.progress_tracker.get_page_id("stories", story_key)
                logger.info(f"⏭️ Skipping existing story: {story_name} ({page_id})")
                return page_id
            
            cluster_metadata = cluster_data.get("cluster_metadata", {})
            
            properties = {
                "Story Name": {
                    "title": [{"text": {"content": cluster_data.get("cluster_name", "Untitled Story")}}]
                },
                "Story Description": {
                    "rich_text": [{"text": {"content": cluster_data.get("cluster_description", "")[:2000]}}]
                },
                "Chapter": {
                    "relation": [{"id": chapter_page_id}]
                },
                "Voice Memo Count": {
                    "number": cluster_metadata.get("voice_memo_count", 0)
                },
                "Average Quality Score": {
                    "number": cluster_metadata.get("average_quality_coefficient", 0.0)
                },
                "Total Reading Time": {
                    "rich_text": [{"text": {"content": cluster_metadata.get("total_reading_time_pretty", "")}}]
                },
                "Total Audio Length": {
                    "rich_text": [{"text": {"content": cluster_metadata.get("total_audio_length_pretty", "")}}]
                },
                "Time Span Days": {
                    "number": cluster_metadata.get("time_span_days", 0)
                },
                "Time Span Pretty": {
                    "rich_text": [{"text": {"content": cluster_metadata.get("time_span_pretty", "")}}]
                }
            }
            
            # Add optional fields
            if cluster_metadata.get("quality_grade_curved"):
                properties["Quality Grade"] = {"select": {"name": cluster_metadata["quality_grade_curved"]}}
                
            if cluster_metadata.get("first_entry_date"):
                properties["First Entry Date"] = {"date": {"start": cluster_metadata["first_entry_date"]}}
                
            if cluster_metadata.get("latest_entry_date"):
                properties["Latest Entry Date"] = {"date": {"start": cluster_metadata["latest_entry_date"]}}
            
            # Create the page
            page_id = self.stories_service._make_api_call(
                "create_page",
                parent={"database_id": self.stories_db_id},
                properties=properties,
                use_cache=False
            )["id"]
            
            # Save to progress tracker
            self.progress_tracker.save_page_created("stories", story_key, page_id, cluster_metadata)
            
            return page_id
            
        except Exception as e:
            logger.error(f"Failed to create story page: {e}")
            return None
    
    def _process_voice_memos_parallel(self, voice_memos: List[Dict[str, Any]], story_page_id: str, max_workers: int = 3) -> List[str]:
        """
        Process multiple voice memos in parallel for better performance
        
        Args:
            voice_memos: List of voice memo data dictionaries
            story_page_id: ID of the parent story page
            max_workers: Maximum number of worker threads
            
        Returns:
            List of created page IDs (may contain None values for failures)
        """
        if not voice_memos:
            return []
        
        logger.info(f"🧵 Processing {len(voice_memos)} voice memos with {max_workers} threads...")
        
        # Reset counters for this batch
        with self._thread_lock:
            self._processed_count = 0
            self._failed_count = 0
        
        created_page_ids = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VoiceMemo") as executor:
            # Submit all voice memo creation tasks
            future_to_memo = {
                executor.submit(self._create_voice_memo_page_safe, voice_memo_data, story_page_id): voice_memo_data
                for voice_memo_data in voice_memos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_memo):
                voice_memo_data = future_to_memo[future]
                filename = voice_memo_data.get("filename", "unknown")
                
                try:
                    page_id = future.result()
                    if page_id:
                        created_page_ids.append(page_id)
                        
                        with self._thread_lock:
                            self._processed_count += 1
                            logger.info(f"🎙️ [{self._processed_count}/{len(voice_memos)}] Created voice memo: {filename}")
                    else:
                        with self._thread_lock:
                            self._failed_count += 1
                            logger.warning(f"❌ [{self._failed_count}] Failed to create: {filename}")
                            
                except Exception as exc:
                    with self._thread_lock:
                        self._failed_count += 1
                        logger.error(f"❌ [{self._failed_count}] Thread error for {filename}: {exc}")
        
        # Final summary for this batch
        successful = len(created_page_ids)
        failed = len(voice_memos) - successful
        logger.info(f"🎯 Voice memo batch complete: {successful} successful, {failed} failed")
        
        return created_page_ids
    
    def _create_voice_memo_page_safe(self, voice_memo_data: Dict[str, Any], story_page_id: str) -> Optional[str]:
        """
        Thread-safe wrapper for _create_voice_memo_page
        Ensures each thread has independent error handling
        """
        try:
            return self._create_voice_memo_page(voice_memo_data, story_page_id)
        except Exception as e:
            filename = voice_memo_data.get("filename", "unknown")
            logger.error(f"❌ Thread-safe error creating voice memo {filename}: {e}")
            return None
    
    def _create_voice_memo_page(self, voice_memo_data: Dict[str, Any], story_page_id: str) -> Optional[str]:
        """Create a voice memo page in the Voice Memos database with full transcript formatting"""
        try:
            filename = voice_memo_data.get("filename", "")
            
            # Check if voice memo already exists
            if self.progress_tracker.is_page_created("voice_memos", filename):
                page_id = self.progress_tracker.get_page_id("voice_memos", filename)
                logger.info(f"⏭️ Skipping existing voice memo: {filename} ({page_id})")
                
                # Still check if audio needs to be uploaded
                if not self.progress_tracker.is_audio_uploaded(filename):
                    audio_file_path = self.get_audio_file_path(filename)
                    if audio_file_path:
                        logger.info(f"📎 Uploading missing audio for existing page: {filename}")
                        success = self._upload_audio_with_retry(page_id, audio_file_path, filename)
                        if success:
                            self.progress_tracker.save_audio_upload_success(filename, page_id)
                        else:
                            self.progress_tracker.save_audio_upload_failure(filename, page_id, "Failed on existing page")
                
                return page_id
            
            transcript = voice_memo_data.get("transcript", "")
            fingerprint = voice_memo_data.get("fingerprint", {})
            metadata = voice_memo_data.get("metadata", {})
            
            # Get Claude processed data if available
            claude_data = self.get_claude_data_for_filename(filename)
            
            # Use Claude title if available, otherwise extract from filename
            if claude_data and claude_data.get("title"):
                title = claude_data["title"]
            else:
                title = filename.replace(".json", "").split("/")[-1] if filename else "Untitled Voice Memo"
            
            properties = {
                "Title": {
                    "title": [{"text": {"content": title}}]
                },
                "Story": {
                    "relation": [{"id": story_page_id}]
                },
                "Filename": {
                    "rich_text": [{"text": {"content": filename}}]
                },
                "Word Count": {
                    "number": metadata.get("word_count", 0)
                },
                "File Size Bytes": {
                    "number": metadata.get("file_size_bytes", 0)
                },
                "Duration (Seconds)": {
                    "number": metadata.get("audio_length_seconds", 0)
                },
                "Audio Length Pretty": {
                    "rich_text": [{"text": {"content": metadata.get("audio_length_pretty", "")}}]
                },
                "Reading Time Pretty": {
                    "rich_text": [{"text": {"content": metadata.get("estimated_reading_time_pretty", "")}}]
                },
                "Quality Score": {
                    "number": metadata.get("quality_coefficient", 0.0)
                }
            }
            
            # Add Claude processed Tags if available
            if claude_data and claude_data.get("entity_extraction"):
                properties["Tags"] = {
                    "rich_text": [{"text": {"content": claude_data["entity_extraction"]}}]
                }
            
            # Add optional fields
            if metadata.get("quality_grade_curved"):
                properties["Quality Grade"] = {"select": {"name": metadata["quality_grade_curved"]}}
                
            if metadata.get("creation_date"):
                properties["Creation Date"] = {"date": {"start": metadata["creation_date"]}}
            
            # Add fingerprint fields
            if fingerprint:
                core_exploration = fingerprint.get("core_exploration", {})
                if core_exploration.get("central_question"):
                    properties["Central Question"] = {
                        "rich_text": [{"text": {"content": core_exploration["central_question"][:2000]}}]
                    }
                if core_exploration.get("key_tension"):
                    properties["Key Tension"] = {
                        "rich_text": [{"text": {"content": core_exploration["key_tension"][:2000]}}]
                    }
                if core_exploration.get("breakthrough_moment"):
                    properties["Breakthrough Moment"] = {
                        "rich_text": [{"text": {"content": core_exploration["breakthrough_moment"][:2000]}}]
                    }
                if core_exploration.get("edge_of_understanding"):
                    properties["Edge of Understanding"] = {
                        "rich_text": [{"text": {"content": core_exploration["edge_of_understanding"][:2000]}}]
                    }
                
                # Pattern fields
                insight_pattern = fingerprint.get("insight_pattern", {})
                if insight_pattern.get("thinking_styles"):
                    properties["Thinking Styles"] = {
                        "multi_select": [{"name": style} for style in insight_pattern["thinking_styles"]]
                    }
                if insight_pattern.get("insight_type"):
                    properties["Insight Type"] = {"select": {"name": insight_pattern["insight_type"]}}
                if insight_pattern.get("development_stage"):
                    properties["Development Stage"] = {"select": {"name": insight_pattern["development_stage"]}}
                if insight_pattern.get("connected_domains"):
                    properties["Connected Domains"] = {
                        "multi_select": [{"name": domain} for domain in insight_pattern["connected_domains"]]
                    }
                
                # Quality scores
                insight_quality = fingerprint.get("insight_quality", {})
                if insight_quality.get("uniqueness_score") is not None:
                    properties["Uniqueness Score"] = {"number": insight_quality["uniqueness_score"]}
                if insight_quality.get("depth_score") is not None:
                    properties["Depth Score"] = {"number": insight_quality["depth_score"]}
                if insight_quality.get("generative_score") is not None:
                    properties["Generative Score"] = {"number": insight_quality["generative_score"]}
                if insight_quality.get("usefulness_score") is not None:
                    properties["Usefulness Score"] = {"number": insight_quality["usefulness_score"]}
                if insight_quality.get("confidence_score") is not None:
                    properties["Confidence Score"] = {"number": insight_quality["confidence_score"]}
                
                # Raw essence and conceptual DNA
                if fingerprint.get("raw_essence"):
                    properties["Raw Essence"] = {
                        "rich_text": [{"text": {"content": fingerprint["raw_essence"][:2000]}}]
                    }
                
                if fingerprint.get("conceptual_dna"):
                    dna_text = "\n".join([f"• {concept}" for concept in fingerprint["conceptual_dna"]])
                    properties["Conceptual DNA"] = {
                        "rich_text": [{"text": {"content": dna_text[:2000]}}]
                    }
            
            # Create page with transcript content
            children = []
            
            # Use Claude formatted text if available, otherwise use raw transcript
            content_to_use = transcript
            if claude_data and claude_data.get("notion_formatted_text"):
                content_to_use = claude_data["notion_formatted_text"]
                logger.debug(f"Using Claude formatted content for {filename}")
            elif transcript:
                logger.debug(f"Using raw transcript for {filename}")
            
            if content_to_use:
                # Use existing markdown conversion functionality
                try:
                    children = self.voice_memos_service._markdown_to_notion_blocks(content_to_use)
                except Exception as e:
                    logger.warning(f"Failed to convert transcript markdown: {e}")
                    # Fallback to plain text
                    children = [{
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": content_to_use[:2000]}}]
                        }
                    }]
            
            # Create the page
            page_id = self.voice_memos_service._make_api_call(
                "create_page",
                parent={"database_id": self.voice_memos_db_id},
                properties=properties,
                children=children,
                use_cache=False
            )["id"]
            
            # Add audio file to the page (if audio file exists)
            audio_file_path = self.get_audio_file_path(filename)
            if audio_file_path:
                # Check if audio was already uploaded
                if self.progress_tracker.is_audio_uploaded(filename):
                    logger.info(f"⏭️ Audio already uploaded for {filename}")
                else:
                    logger.info(f"📎 Uploading audio file for {filename}")
                    success = self._upload_audio_with_retry(page_id, audio_file_path, filename)
                    if success:
                        logger.info(f"✅ Successfully uploaded audio file for {filename}")
                        self.progress_tracker.save_audio_upload_success(filename, page_id)
                    else:
                        logger.error(f"❌ Failed to upload audio file for {filename} after retries")
                        self.progress_tracker.save_audio_upload_failure(filename, page_id, "Failed after max retries")
            else:
                logger.warning(f"⚠️ Audio file not found for {filename}, skipping upload")
            
            # Save voice memo creation to progress
            self.progress_tracker.save_page_created("voice_memos", filename, page_id, {"title": title})
            
            return page_id
            
        except Exception as e:
            logger.error(f"Failed to create voice memo page: {e}")
            return None
    
    def setup_filtered_views(self) -> Dict[str, List[str]]:
        """
        Setup filtered database views for hierarchical navigation
        This creates linked database blocks that show filtered content based on relations
        
        Returns:
            Dictionary mapping view types to their block IDs
        """
        try:
            logger.info("🔍 Setting up filtered views for hierarchical navigation...")
            
            created_views = {
                "book_chapter_views": [],
                "book_story_views": [],
                "book_voice_memo_views": [],
                "chapter_story_views": [],
                "chapter_voice_memo_views": [],
                "story_voice_memo_views": []
            }
            
            # Note: Notion API doesn't directly support creating filtered database views programmatically
            # However, we can create linked database blocks with filters in the page content
            # This would typically be done through the UI or by adding blocks to existing pages
            
            logger.info("ℹ️ Filtered views need to be configured manually in Notion UI")
            logger.info("💡 Suggested manual setup:")
            logger.info("   1. On each Book page, add linked database blocks for Chapters, Stories, and Voice Memos")
            logger.info("   2. Filter Chapters by Book relation")
            logger.info("   3. Filter Stories by Book rollup")
            logger.info("   4. Filter Voice Memos by Book rollup")
            logger.info("   5. Repeat similar filtering for Chapter pages")
            
            return created_views
            
        except Exception as e:
            logger.error(f"❌ Failed to setup filtered views: {e}")
            raise
    
    def create_navigation_guide(self) -> str:
        """
        Create a navigation guide page that explains the hierarchical structure
        
        Returns:
            Page ID of the created navigation guide
        """
        try:
            logger.info("📋 Creating navigation guide...")
            
            # Create a temporary service for the parent page
            temp_service = NotionService(database_id="temp", notion_token=self.notion_token)
            
            # Navigation guide content
            children = [
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": "🗺️ Hidden Genius Library Navigation Guide"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"text": {"content": "This library is organized into four interconnected databases that mirror your voice memo collection structure."}}
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "📚 Database Structure"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {"text": {"content": "Books ", "annotations": {"bold": True}}},
                            {"text": {"content": "- Top-level categories (Business, Spirituality, etc.)"}}  
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {"text": {"content": "Chapters ", "annotations": {"bold": True}}},
                            {"text": {"content": "- Thematic sections within each book"}}  
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {"text": {"content": "Stories ", "annotations": {"bold": True}}},
                            {"text": {"content": "- Related voice memo clusters (renamed from 'clusters')"}}  
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {"text": {"content": "Voice Memos ", "annotations": {"bold": True}}},
                            {"text": {"content": "- Individual voice recordings with full transcripts and analysis"}}  
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "🔗 Database Links"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"text": {"content": "📚 Books Database: "}},
                            {
                                "text": {"content": f"https://www.notion.so/{self.books_db_id.replace('-', '')}"},
                                "href": f"https://www.notion.so/{self.books_db_id.replace('-', '')}"
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"text": {"content": "📖 Chapters Database: "}},
                            {
                                "text": {"content": f"https://www.notion.so/{self.chapters_db_id.replace('-', '')}"},
                                "href": f"https://www.notion.so/{self.chapters_db_id.replace('-', '')}"
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"text": {"content": "📝 Stories Database: "}},
                            {
                                "text": {"content": f"https://www.notion.so/{self.stories_db_id.replace('-', '')}"},
                                "href": f"https://www.notion.so/{self.stories_db_id.replace('-', '')}"
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"text": {"content": "🎙️ Voice Memos Database: "}},
                            {
                                "text": {"content": f"https://www.notion.so/{self.voice_memos_db_id.replace('-', '')}"},
                                "href": f"https://www.notion.so/{self.voice_memos_db_id.replace('-', '')}"
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "🎯 How to Navigate"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "Start at the Books level to explore high-level topics"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "Click into a Book page to see filtered views of its Chapters, Stories, and Voice Memos"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "Use the relation properties to navigate between connected items"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "Voice Memos contain the full transcripts with beautiful formatting and rich metadata"}}]
                    }
                }
            ]
            
            # Create the page
            response = temp_service._make_api_call(
                "create_page",
                parent={"page_id": self.parent_page_id},
                properties={
                    "title": {
                        "title": [{"text": {"content": "🗺️ Library Navigation Guide"}}]
                    }
                },
                children=children,
                use_cache=False
            )
            
            guide_page_id = response["id"]
            logger.info(f"✅ Navigation guide created: {guide_page_id}")
            
            return guide_page_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create navigation guide: {e}")
            raise
    
    def get_database_ids(self) -> Dict[str, str]:
        """
        Get the database IDs for external reference
        
        Returns:
            Dictionary mapping database names to their IDs
        """
        return {
            "books": self.books_db_id,
            "chapters": self.chapters_db_id,
            "stories": self.stories_db_id,
            "voice_memos": self.voice_memos_db_id
        }
    
    def retry_failed_audio_uploads(self, max_additional_retries: int = 2) -> Dict[str, bool]:
        """
        Retry all failed audio uploads with longer timeouts
        
        Args:
            max_additional_retries: Number of additional retry attempts
            
        Returns:
            Dictionary mapping filenames to success status
        """
        failed_uploads = self.progress_tracker.get_failed_uploads()
        
        if not failed_uploads:
            logger.info("✅ No failed audio uploads to retry")
            return {}
        
        logger.info(f"🔄 Retrying {len(failed_uploads)} failed audio uploads...")
        results = {}
        
        for filename, failure_data in failed_uploads.items():
            page_id = failure_data.get("page_id")
            audio_file_path = self.get_audio_file_path(filename)
            
            if not audio_file_path:
                logger.warning(f"⚠️ Audio file not found for retry: {filename}")
                results[filename] = False
                continue
            
            logger.info(f"🔄 Retrying upload for {filename} (previous attempts: {failure_data.get('attempts', 0)})")
            
            # Try with longer timeouts between retries
            for attempt in range(max_additional_retries):
                try:
                    logger.info(f"📤 Retry attempt {attempt + 1}/{max_additional_retries} for {filename}")
                    
                    success = self.voice_memos_service.add_audio_file_to_page(page_id, audio_file_path)
                    
                    if success:
                        logger.info(f"✅ Successfully uploaded {filename} on retry!")
                        self.progress_tracker.save_audio_upload_success(filename, page_id)
                        results[filename] = True
                        break
                    
                    # Longer wait time for retries
                    if attempt < max_additional_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5, 10 seconds
                        logger.warning(f"⏳ Upload failed, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        results[filename] = False
                        
                except Exception as e:
                    logger.error(f"❌ Retry attempt {attempt + 1} failed: {e}")
                    self.progress_tracker.save_audio_upload_failure(
                        filename, page_id, f"Retry attempt {attempt + 1}: {str(e)}"
                    )
                    
                    if attempt < max_additional_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        time.sleep(wait_time)
                    else:
                        results[filename] = False
        
        # Report final results
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        
        logger.info(f"📊 Retry results: {successful} successful, {failed} failed")
        
        if failed > 0:
            logger.warning(f"⚠️ {failed} audio files still failed after all retries:")
            for filename, success in results.items():
                if not success:
                    logger.warning(f"  - {filename}")
        
        return results
    
    def rebuild_progress_tracker_from_notion(self):
        """
        Rebuild the progress tracker by querying existing pages in Notion databases.
        This fixes cases where pages exist but weren't tracked properly.
        """
        logger.info("🔄 Rebuilding progress tracker from existing Notion pages...")
        
        try:
            # Query chapters database
            if self.chapters_service and self.chapters_db_id:
                logger.info("📖 Querying existing chapters...")
                chapters_response = self.chapters_service._make_api_call(
                    "query_database",
                    database_id=self.chapters_db_id,
                    use_cache=False
                )
                
                for page in chapters_response.get("results", []):
                    chapter_title = ""
                    part_title = ""
                    
                    # Extract title and part from properties
                    props = page.get("properties", {})
                    if "Chapter Title" in props and props["Chapter Title"]["title"]:
                        chapter_title = props["Chapter Title"]["title"][0]["text"]["content"]
                    if "Part Title" in props and props["Part Title"]["rich_text"]:
                        part_title = props["Part Title"]["rich_text"][0]["text"]["content"] if props["Part Title"]["rich_text"] else ""
                    
                    # Create the same key format used in creation
                    chapter_key = f"{part_title}/{chapter_title}" if part_title else chapter_title
                    
                    # Save to progress tracker if not already tracked
                    if not self.progress_tracker.is_page_created("chapters", chapter_key):
                        self.progress_tracker.save_page_created("chapters", chapter_key, page["id"])
                        logger.info(f"📖 Tracked existing chapter: {chapter_key}")
            
            # Query stories database  
            if self.stories_service and self.stories_db_id:
                logger.info("📝 Querying existing stories...")
                stories_response = self.stories_service._make_api_call(
                    "query_database",
                    database_id=self.stories_db_id,
                    use_cache=False
                )
                
                for page in stories_response.get("results", []):
                    story_name = ""
                    chapter_id = ""
                    
                    # Extract story name and chapter relation
                    props = page.get("properties", {})
                    if "Story Name" in props and props["Story Name"]["title"]:
                        story_name = props["Story Name"]["title"][0]["text"]["content"]
                    if "Chapter" in props and props["Chapter"]["relation"]:
                        chapter_id = props["Chapter"]["relation"][0]["id"] if props["Chapter"]["relation"] else ""
                    
                    # Create the same key format used in creation
                    story_key = f"{chapter_id}/{story_name}"
                    
                    # Save to progress tracker if not already tracked
                    if not self.progress_tracker.is_page_created("stories", story_key):
                        self.progress_tracker.save_page_created("stories", story_key, page["id"])
                        logger.info(f"📝 Tracked existing story: {story_name}")
            
            logger.info("✅ Progress tracker rebuilt from existing Notion pages")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild progress tracker: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get combined performance statistics from all database services
        
        Returns:
            Dictionary with performance statistics from all services
        """
        stats = {}
        
        if self.books_service:
            stats["books"] = self.books_service.get_performance_stats()
        if self.chapters_service:
            stats["chapters"] = self.chapters_service.get_performance_stats()
        if self.stories_service:
            stats["stories"] = self.stories_service.get_performance_stats()
        if self.voice_memos_service:
            stats["voice_memos"] = self.voice_memos_service.get_performance_stats()
        
        # Calculate totals
        total_api_calls = sum(s.get("api_calls_made", 0) for s in stats.values())
        total_cache_hits = sum(s.get("cache_hits", 0) for s in stats.values())
        total_cache_misses = sum(s.get("cache_misses", 0) for s in stats.values())
        total_cached_items = sum(s.get("cached_items", 0) for s in stats.values())
        
        cache_total = total_cache_hits + total_cache_misses
        overall_hit_rate = (total_cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        stats["totals"] = {
            "total_api_calls": total_api_calls,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "overall_cache_hit_rate_percent": round(overall_hit_rate, 2),
            "total_cached_items": total_cached_items,
            "estimated_total_cost_savings": total_cache_hits * 0.01
        }
        
        return stats