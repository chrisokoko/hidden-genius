#!/usr/bin/env python3
"""
Assign Voice Memos to Stories in Notion

This script:
1. Queries the Voice Memos database to find entries without a Story relation
2. Extracts the filename/title from each unattached memo
3. Looks up the appropriate cluster/story from clustering data
4. Updates the Voice Memo to link it to the corresponding Story in Notion
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from notion_service import NotionService
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceMemoStoryAssigner:
    """Assigns unattached voice memos to their appropriate stories in Notion"""
    
    def __init__(self, 
                 voice_memos_db_id: str,
                 stories_db_id: str,
                 notion_token: str = None,
                 enriched_clusters_file: str = "data/05 filtered_enriched_clusters/filtered_coarse_clusters_condensed.json",
                 chapter_assignment_file: str = "data/07 books/chapter_cluster_assigment.json"):
        """
        Initialize the assigner
        
        Args:
            voice_memos_db_id: ID of the Voice Memos database in Notion
            stories_db_id: ID of the Stories database in Notion
            notion_token: Notion API token (will use env var if not provided)
            enriched_clusters_file: Path to the enriched clusters JSON file
            chapter_assignment_file: Path to the chapter cluster assignment JSON file
        """
        self.notion_token = notion_token or os.getenv('NOTION_TOKEN')
        if not self.notion_token:
            raise ValueError("NOTION_TOKEN not provided and not found in environment")
        
        self.voice_memos_db_id = voice_memos_db_id
        self.stories_db_id = stories_db_id
        
        # Initialize Notion services
        self.voice_memos_service = NotionService(voice_memos_db_id, self.notion_token)
        self.stories_service = NotionService(stories_db_id, self.notion_token)
        
        # Load enriched clusters and chapter assignments
        self.enriched_clusters = self._load_enriched_clusters(enriched_clusters_file)
        self.chapter_assignments = self._load_chapter_assignments(chapter_assignment_file)
        
        # Build lookup maps
        self.filename_to_cluster = self._build_filename_cluster_map()
        self.cluster_to_book_chapter = self._build_cluster_to_book_chapter_map()
        self.story_lookup = {}  # Will be populated when querying stories
        
        # Statistics
        self.stats = {
            'total_memos': 0,
            'unattached_memos': 0,
            'successfully_assigned': 0,
            'failed_assignments': 0,
            'no_cluster_found': 0,
            'no_story_found': 0
        }
    
    def _load_enriched_clusters(self, enriched_clusters_file: str) -> Optional[Dict[str, Any]]:
        """Load the enriched clusters data"""
        try:
            with open(enriched_clusters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded enriched clusters from {enriched_clusters_file}")
                return data
        except FileNotFoundError:
            logger.error(f"❌ Enriched clusters file not found: {enriched_clusters_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in enriched clusters file: {e}")
            return None
    
    def _load_chapter_assignments(self, chapter_assignment_file: str) -> Optional[List[Dict[str, Any]]]:
        """Load the chapter cluster assignments"""
        try:
            with open(chapter_assignment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded chapter assignments from {chapter_assignment_file}")
                return data
        except FileNotFoundError:
            logger.error(f"❌ Chapter assignment file not found: {chapter_assignment_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in chapter assignment file: {e}")
            return None
    
    def _build_filename_cluster_map(self) -> Dict[str, int]:
        """Build a map from filename to cluster number"""
        filename_map = {}
        
        if not self.enriched_clusters:
            logger.warning("No enriched clusters data available")
            return filename_map
        
        # Extract clusters from enriched data
        clusters = self.enriched_clusters.get('clusters', {})
        
        # Iterate through each cluster
        for cluster_id, cluster_data in clusters.items():
            voice_memos = cluster_data.get('voice_memos', [])
            
            # Map each voice memo filename to this cluster
            for memo in voice_memos:
                filename = memo.get('filename', '')
                if filename:
                    # Clean the filename (remove .json extension if present)
                    clean_filename = filename.replace('.json', '')
                    
                    # Store the cluster ID (convert to int)
                    filename_map[clean_filename] = int(cluster_id)
                    
                    # Also store without path prefix
                    base_filename = Path(clean_filename).name
                    if base_filename != clean_filename:
                        filename_map[base_filename] = int(cluster_id)
        
        logger.info(f"Built filename-cluster map with {len(filename_map)} entries")
        return filename_map
    
    def _build_cluster_to_book_chapter_map(self) -> Dict[int, Dict[str, str]]:
        """Build a map from cluster ID to book/chapter/part information"""
        cluster_map = {}
        
        if not self.chapter_assignments:
            logger.warning("No chapter assignments available")
            return cluster_map
        
        # Iterate through each book
        for book in self.chapter_assignments:
            book_title = book.get('book_title', '')
            chapters = book.get('chapters', [])
            
            # Iterate through each chapter
            for chapter in chapters:
                chapter_title = chapter.get('chapter_title', '')
                part_title = chapter.get('part_title', '')
                cluster_ids = chapter.get('clusters', [])
                
                # Map each cluster ID to this book/chapter/part
                for cluster_id in cluster_ids:
                    cluster_map[cluster_id] = {
                        'book_title': book_title,
                        'part_title': part_title,
                        'chapter_title': chapter_title,
                        'full_path': f"{book_title} > {part_title} > {chapter_title}"
                    }
        
        logger.info(f"Built cluster-to-book-chapter map with {len(cluster_map)} entries")
        return cluster_map
    
    def query_unattached_voice_memos(self) -> List[Dict[str, Any]]:
        """Query all voice memos that don't have a Story relation"""
        logger.info("🔍 Querying voice memos database...")
        
        try:
            # Query all voice memos
            all_memos = self.voice_memos_service.query_all_pages()
            self.stats['total_memos'] = len(all_memos)
            
            unattached_memos = []
            
            for memo in all_memos:
                properties = memo.get('properties', {})
                
                # Check if Story relation is empty
                story_relation = properties.get('Story', {})
                
                # The relation field will have an empty 'relation' array if unattached
                if story_relation.get('type') == 'relation':
                    relations = story_relation.get('relation', [])
                    if not relations:  # Empty relation means unattached
                        unattached_memos.append(memo)
            
            self.stats['unattached_memos'] = len(unattached_memos)
            logger.info(f"Found {len(unattached_memos)} unattached voice memos out of {len(all_memos)} total")
            
            return unattached_memos
            
        except Exception as e:
            logger.error(f"Error querying voice memos: {e}")
            return []
    
    def query_all_stories(self) -> Dict[str, str]:
        """Query all stories and build a lookup map from cluster ID to story page ID"""
        logger.info("📝 Querying stories database...")
        
        try:
            all_stories = self.stories_service.query_all_pages()
            
            for story in all_stories:
                properties = story.get('properties', {})
                story_id = story.get('id')
                
                # Get cluster ID from the story
                cluster_id_prop = properties.get('Cluster ID', {})
                if cluster_id_prop.get('type') == 'rich_text':
                    rich_text = cluster_id_prop.get('rich_text', [])
                    if rich_text and rich_text[0].get('text'):
                        cluster_id = rich_text[0]['text']['content']
                        self.story_lookup[cluster_id] = story_id
                
                # Also try to get story name for additional matching
                story_name_prop = properties.get('Story Name', {})
                if story_name_prop.get('type') == 'title':
                    title = story_name_prop.get('title', [])
                    if title and title[0].get('text'):
                        story_name = title[0]['text']['content']
                        # Store by name as well (might be useful for fuzzy matching)
                        self.story_lookup[f"name:{story_name}"] = story_id
            
            logger.info(f"Found {len(self.story_lookup)} stories in database")
            return self.story_lookup
            
        except Exception as e:
            logger.error(f"Error querying stories: {e}")
            return {}
    
    def extract_filename_from_memo(self, memo: Dict[str, Any]) -> Optional[str]:
        """Extract filename from a voice memo page"""
        properties = memo.get('properties', {})
        
        # Try to get filename from Filename property
        filename_prop = properties.get('Filename', {})
        if filename_prop.get('type') == 'rich_text':
            rich_text = filename_prop.get('rich_text', [])
            if rich_text and rich_text[0].get('text'):
                return rich_text[0]['text']['content']
        
        # Fallback to title
        title_prop = properties.get('Title', {})
        if title_prop.get('type') == 'title':
            title = title_prop.get('title', [])
            if title and title[0].get('text'):
                return title[0]['text']['content']
        
        return None
    
    def find_cluster_for_filename(self, filename: str) -> Optional[int]:
        """Find cluster ID for a given filename"""
        if not filename:
            return None
        
        # Clean the filename
        clean_filename = filename.replace('.json', '').replace('.txt', '')
        
        # Try exact match
        if clean_filename in self.filename_to_cluster:
            return self.filename_to_cluster[clean_filename]
        
        # Try just the base filename
        base_filename = Path(clean_filename).name
        if base_filename in self.filename_to_cluster:
            return self.filename_to_cluster[base_filename]
        
        # Try with different extensions
        for ext in ['.json', '.txt', '']:
            test_name = f"{clean_filename}{ext}"
            if test_name in self.filename_to_cluster:
                return self.filename_to_cluster[test_name]
        
        return None
    
    def find_book_chapter_for_cluster(self, cluster_id: int) -> Optional[Dict[str, str]]:
        """Find book/chapter/part information for a given cluster ID"""
        return self.cluster_to_book_chapter.get(cluster_id)
    
    def find_story_for_book_chapter(self, book_chapter_info: Dict[str, str]) -> Optional[str]:
        """Find the story page ID for a given book/chapter combination"""
        # Try to match by chapter title first (most specific)
        chapter_title = book_chapter_info.get('chapter_title', '')
        
        for story_name, story_id in self.story_lookup.items():
            if story_name.startswith('name:'):
                actual_name = story_name[5:]  # Remove 'name:' prefix
                # Try to match chapter title in story name
                if chapter_title and chapter_title.lower() in actual_name.lower():
                    return story_id
        
        # If no match, try to match by cluster ID directly
        # This requires updating the story lookup to include cluster IDs
        return None
    
    def update_voice_memo_story(self, memo_id: str, story_id: str) -> bool:
        """Update a voice memo to link it to a story"""
        try:
            properties = {
                "Story": {
                    "relation": [
                        {"id": story_id}
                    ]
                }
            }
            
            success = self.voice_memos_service.update_page(memo_id, properties)
            
            if success:
                logger.info(f"✅ Successfully linked memo to story")
                return True
            else:
                logger.error(f"❌ Failed to update memo")
                return False
                
        except Exception as e:
            logger.error(f"Error updating memo {memo_id}: {e}")
            return False
    
    def process_unattached_memos(self):
        """Main process to find and assign unattached memos to stories"""
        logger.info("=" * 60)
        logger.info("Starting Voice Memo to Story Assignment Process")
        logger.info("=" * 60)
        
        # Step 1: Query all stories to build lookup
        self.query_all_stories()
        
        if not self.story_lookup:
            logger.error("❌ No stories found in database. Please ensure stories are created first.")
            return
        
        # Step 2: Query unattached voice memos
        unattached_memos = self.query_unattached_voice_memos()
        
        if not unattached_memos:
            logger.info("✅ No unattached voice memos found. All memos are already assigned!")
            return
        
        # Step 3: Process each unattached memo
        logger.info(f"\n🔄 Processing {len(unattached_memos)} unattached memos...")
        
        for i, memo in enumerate(unattached_memos, 1):
            memo_id = memo.get('id')
            
            # Extract filename
            filename = self.extract_filename_from_memo(memo)
            
            if not filename:
                logger.warning(f"  [{i}/{len(unattached_memos)}] ⚠️ No filename found for memo {memo_id}")
                continue
            
            logger.info(f"  [{i}/{len(unattached_memos)}] Processing: {filename}")
            
            # Step 1: Find cluster ID for this filename
            cluster_id = self.find_cluster_for_filename(filename)
            
            if cluster_id is None:
                logger.warning(f"    ❌ No cluster found for {filename}")
                self.stats['no_cluster_found'] += 1
                continue
            
            logger.info(f"    📊 Found cluster ID: {cluster_id}")
            
            # Step 2: Find book/chapter for this cluster
            book_chapter_info = self.find_book_chapter_for_cluster(cluster_id)
            
            if not book_chapter_info:
                logger.warning(f"    ❌ No book/chapter mapping for cluster {cluster_id}")
                self.stats['no_story_found'] += 1
                continue
            
            logger.info(f"    📚 Found book/chapter: {book_chapter_info['full_path']}")
            
            # Step 3: Find story for this book/chapter (or cluster ID)
            # First try to find by cluster ID directly
            story_id = self.story_lookup.get(str(cluster_id))
            
            if not story_id:
                # Try to find by book/chapter info
                story_id = self.find_story_for_book_chapter(book_chapter_info)
            
            if not story_id:
                logger.warning(f"    ❌ No story found for {book_chapter_info['full_path']}")
                self.stats['no_story_found'] += 1
                continue
            
            # Update the memo with the story relation
            if self.update_voice_memo_story(memo_id, story_id):
                self.stats['successfully_assigned'] += 1
            else:
                self.stats['failed_assignments'] += 1
            
            # Rate limiting
            time.sleep(0.1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Total voice memos in database: {self.stats['total_memos']}")
        logger.info(f"🔍 Unattached memos found: {self.stats['unattached_memos']}")
        logger.info(f"✅ Successfully assigned: {self.stats['successfully_assigned']}")
        logger.info(f"❌ Failed assignments: {self.stats['failed_assignments']}")
        logger.info(f"❓ No cluster found: {self.stats['no_cluster_found']}")
        logger.info(f"📝 No story found: {self.stats['no_story_found']}")
        
        success_rate = 0
        if self.stats['unattached_memos'] > 0:
            success_rate = (self.stats['successfully_assigned'] / self.stats['unattached_memos']) * 100
            logger.info(f"\n🎯 Success rate: {success_rate:.1f}%")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Assign unattached voice memos to stories in Notion')
    parser.add_argument('--voice-memos-db', required=True, help='Voice Memos database ID')
    parser.add_argument('--stories-db', required=True, help='Stories database ID')
    parser.add_argument('--token', help='Notion API token (or set NOTION_TOKEN env var)')
    parser.add_argument('--enriched-clusters', 
                       default='data/05 filtered_enriched_clusters/filtered_coarse_clusters_condensed.json',
                       help='Path to enriched clusters file')
    parser.add_argument('--chapter-assignments',
                       default='data/07 books/chapter_cluster_assigment.json',
                       help='Path to chapter cluster assignment file')
    
    args = parser.parse_args()
    
    try:
        # Create assigner
        assigner = VoiceMemoStoryAssigner(
            voice_memos_db_id=args.voice_memos_db,
            stories_db_id=args.stories_db,
            notion_token=args.token,
            enriched_clusters_file=args.enriched_clusters,
            chapter_assignment_file=args.chapter_assignments
        )
        
        # Process unattached memos
        assigner.process_unattached_memos()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())