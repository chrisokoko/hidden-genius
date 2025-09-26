#!/usr/bin/env python3
"""
Analyze Voice Memo to Story Mappings

This script analyzes unattached voice memos and generates a mapping report
showing which story each memo would be assigned to, without making any changes.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from notion_service import NotionService
from datetime import datetime
from collections import defaultdict
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceMemoMappingAnalyzer:
    """Analyzes potential voice memo to story mappings without making changes"""
    
    def __init__(self, 
                 voice_memos_db_id: str,
                 stories_db_id: str,
                 notion_token: str = None,
                 enriched_clusters_file: str = "data/05 filtered_enriched_clusters/filtered_medium_clusters_condensed.json",
                 chapter_assignment_file: str = "data/07 books/chapter_cluster_assigment.json",
                 cluster_descriptions_file: str = "data/06 cluster_descriptions/medium.json"):
        """Initialize the analyzer"""
        self.notion_token = notion_token or os.getenv('NOTION_TOKEN')
        if not self.notion_token:
            raise ValueError("NOTION_TOKEN not provided and not found in environment")
        
        self.voice_memos_db_id = voice_memos_db_id
        self.stories_db_id = stories_db_id
        
        # Initialize Notion services
        self.voice_memos_service = NotionService(voice_memos_db_id, self.notion_token)
        self.stories_service = NotionService(stories_db_id, self.notion_token)
        
        # Load enriched clusters, chapter assignments, and descriptions
        self.enriched_clusters = self._load_enriched_clusters(enriched_clusters_file)
        self.chapter_assignments = self._load_chapter_assignments(chapter_assignment_file)
        self.cluster_descriptions = self._load_cluster_descriptions(cluster_descriptions_file)
        
        # Build lookup maps
        self.filename_to_cluster = self._build_filename_cluster_map()
        self.cluster_to_book_chapter = self._build_cluster_to_book_chapter_map()
        self.cluster_to_description = self._build_cluster_to_description_map()
        
        # Will be populated when querying
        self.existing_stories = {}
        self.unattached_memos = []
        self.proposed_mappings = []
        self.stories_to_create = set()
    
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
    
    def _load_cluster_descriptions(self, cluster_descriptions_file: str) -> Optional[Dict[str, Any]]:
        """Load cluster descriptions"""
        try:
            with open(cluster_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded cluster descriptions from {cluster_descriptions_file}")
                return data
        except FileNotFoundError:
            logger.warning(f"⚠️ Cluster descriptions file not found: {cluster_descriptions_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in cluster descriptions file: {e}")
            return None
    
    def _build_filename_cluster_map(self) -> Dict[str, int]:
        """Build a map from filename to cluster number"""
        filename_map = {}
        
        if not self.enriched_clusters:
            logger.warning("No enriched clusters data available")
            return filename_map
        
        clusters = self.enriched_clusters.get('clusters', {})
        
        for cluster_id, cluster_data in clusters.items():
            voice_memos = cluster_data.get('voice_memos', [])
            
            for memo in voice_memos:
                filename = memo.get('filename', '')
                if filename:
                    clean_filename = filename.replace('.json', '')
                    filename_map[clean_filename] = int(cluster_id)
                    
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
        
        for book in self.chapter_assignments:
            book_title = book.get('book_title', '')
            chapters = book.get('chapters', [])
            
            for chapter in chapters:
                chapter_title = chapter.get('chapter_title', '')
                part_title = chapter.get('part_title', '')
                cluster_ids = chapter.get('clusters', [])
                
                for cluster_id in cluster_ids:
                    cluster_map[cluster_id] = {
                        'book_title': book_title,
                        'part_title': part_title,
                        'chapter_title': chapter_title,
                        'full_path': f"{book_title} > {part_title} > {chapter_title}"
                    }
        
        logger.info(f"Built cluster-to-book-chapter map with {len(cluster_map)} entries")
        return cluster_map
    
    def _build_cluster_to_description_map(self) -> Dict[int, Dict[str, str]]:
        """Build a map from cluster ID to description"""
        description_map = {}
        
        if not self.cluster_descriptions:
            logger.warning("No cluster descriptions available")
            return description_map
        
        clusters = self.cluster_descriptions.get('clusters', [])
        
        for cluster in clusters:
            cluster_id = int(cluster.get('cluster_id', 0))
            title = cluster.get('title', '')
            raw_essence = cluster.get('raw_essence', '')
            
            if cluster_id:
                description_map[cluster_id] = {
                    'title': title,
                    'raw_essence': raw_essence
                }
        
        logger.info(f"Built cluster-to-description map with {len(description_map)} entries")
        return description_map
    
    def query_existing_stories(self) -> Dict[str, Dict[str, Any]]:
        """Query all existing stories in the database"""
        logger.info("📝 Querying existing stories...")
        
        try:
            all_stories = self.stories_service.query_all_pages()
            
            for story in all_stories:
                properties = story.get('properties', {})
                story_id = story.get('id')
                
                # Get story name
                story_name_prop = properties.get('Story Name', {})
                if story_name_prop.get('type') == 'title':
                    title = story_name_prop.get('title', [])
                    if title and title[0].get('text'):
                        story_name = title[0]['text']['content']
                        self.existing_stories[story_name.lower()] = {
                            'id': story_id,
                            'name': story_name
                        }
            
            logger.info(f"Found {len(self.existing_stories)} existing stories")
            return self.existing_stories
            
        except Exception as e:
            logger.error(f"Error querying stories: {e}")
            return {}
    
    def query_unattached_voice_memos(self) -> List[Dict[str, Any]]:
        """Query all voice memos that don't have a Story relation"""
        logger.info("🔍 Querying unattached voice memos...")
        
        try:
            all_memos = self.voice_memos_service.query_all_pages()
            unattached = []
            
            for memo in all_memos:
                properties = memo.get('properties', {})
                
                # Check if Story relation is empty
                story_relation = properties.get('Story', {})
                if story_relation.get('type') == 'relation':
                    relations = story_relation.get('relation', [])
                    if not relations:  # Empty relation means unattached
                        # Extract memo details
                        memo_info = {
                            'id': memo.get('id'),
                            'properties': properties
                        }
                        
                        # Get title
                        title_prop = properties.get('Title', {})
                        if title_prop.get('type') == 'title':
                            title = title_prop.get('title', [])
                            if title and title[0].get('text'):
                                memo_info['title'] = title[0]['text']['content']
                        
                        # Get filename
                        filename_prop = properties.get('Filename', {})
                        if filename_prop.get('type') == 'rich_text':
                            rich_text = filename_prop.get('rich_text', [])
                            if rich_text and rich_text[0].get('text'):
                                memo_info['filename'] = rich_text[0]['text']['content']
                        
                        unattached.append(memo_info)
            
            logger.info(f"Found {len(unattached)} unattached voice memos")
            self.unattached_memos = unattached
            return unattached
            
        except Exception as e:
            logger.error(f"Error querying voice memos: {e}")
            return []
    
    def generate_story_name_for_cluster(self, cluster_id: int) -> str:
        """Generate a descriptive story name for a cluster with ID prefix"""
        # First try to get the description
        cluster_desc = self.cluster_to_description.get(cluster_id)
        
        if cluster_desc and cluster_desc.get('title'):
            # Use format "Story [ID]: [Title]"
            return f"Story {cluster_id}: {cluster_desc['title']}"
        
        # Fallback to book/chapter info
        book_chapter = self.cluster_to_book_chapter.get(cluster_id)
        if book_chapter:
            chapter_title = book_chapter.get('chapter_title', '')
            if chapter_title:
                # Remove "Chapter X: " prefix if present
                if ':' in chapter_title:
                    chapter_name = chapter_title.split(':', 1)[1].strip()
                else:
                    chapter_name = chapter_title
                return f"Story {cluster_id}: {chapter_name}"
        
        # Final fallback
        return f"Story {cluster_id}: Untitled"
    
    def analyze_mappings(self):
        """Analyze all potential mappings without making changes"""
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING VOICE MEMO TO STORY MAPPINGS")
        logger.info("=" * 80)
        
        # Query existing data
        self.query_existing_stories()
        self.query_unattached_voice_memos()
        
        if not self.unattached_memos:
            logger.info("✅ No unattached voice memos found!")
            return
        
        # Analyze each unattached memo
        mappings_by_story = defaultdict(list)
        unmapped_memos = []
        
        for memo in self.unattached_memos:
            filename = memo.get('filename') or memo.get('title', '')
            
            if not filename:
                unmapped_memos.append(memo)
                continue
            
            # Find cluster for this filename
            cluster_id = None
            clean_filename = filename.replace('.json', '').replace('.txt', '')
            
            if clean_filename in self.filename_to_cluster:
                cluster_id = self.filename_to_cluster[clean_filename]
            else:
                base_filename = Path(clean_filename).name
                if base_filename in self.filename_to_cluster:
                    cluster_id = self.filename_to_cluster[base_filename]
            
            if cluster_id is None:
                unmapped_memos.append(memo)
                continue
            
            # Get book/chapter info
            book_chapter = self.cluster_to_book_chapter.get(cluster_id)
            if not book_chapter:
                unmapped_memos.append(memo)
                continue
            
            # Generate story name
            story_name = self.generate_story_name_for_cluster(cluster_id)
            
            # Check if story exists
            story_exists = story_name.lower() in self.existing_stories
            
            mapping = {
                'memo_id': memo['id'],
                'memo_title': memo.get('title', 'Untitled'),
                'filename': filename,
                'cluster_id': cluster_id,
                'book': book_chapter['book_title'],
                'part': book_chapter['part_title'],
                'chapter': book_chapter['chapter_title'],
                'story_name': story_name,
                'story_exists': story_exists
            }
            
            self.proposed_mappings.append(mapping)
            mappings_by_story[story_name].append(mapping)
            
            if not story_exists:
                self.stories_to_create.add(story_name)
        
        # Generate report
        self.generate_report(mappings_by_story, unmapped_memos)
    
    def generate_report(self, mappings_by_story: Dict[str, List[Dict]], unmapped_memos: List[Dict]):
        """Generate a detailed mapping report"""
        
        # Console output
        print("\n" + "=" * 80)
        print("VOICE MEMO TO STORY MAPPING REPORT")
        print("=" * 80)
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Total unattached memos: {len(self.unattached_memos)}")
        print(f"  • Memos with mappings: {len(self.proposed_mappings)}")
        print(f"  • Unmapped memos: {len(unmapped_memos)}")
        print(f"  • Existing stories to use: {len([s for s in mappings_by_story if s.lower() in self.existing_stories])}")
        print(f"  • New stories to create: {len(self.stories_to_create)}")
        
        # Stories to create
        if self.stories_to_create:
            print(f"\n📝 NEW STORIES TO CREATE ({len(self.stories_to_create)}):")
            for story_name in sorted(self.stories_to_create):
                memo_count = len(mappings_by_story[story_name])
                print(f"  • {story_name} ({memo_count} memos)")
        
        # Mappings by story
        print(f"\n🔗 PROPOSED MAPPINGS BY STORY:")
        for story_name in sorted(mappings_by_story.keys()):
            memos = mappings_by_story[story_name]
            exists = " [EXISTS]" if story_name.lower() in self.existing_stories else " [NEW]"
            print(f"\n  📚 {story_name}{exists} ({len(memos)} memos)")
            
            # Group by book/chapter for context
            book_chapter = f"{memos[0]['book']} > {memos[0]['chapter']}"
            print(f"     {book_chapter}")
            
            # List first few memos
            for i, memo in enumerate(memos[:5]):
                print(f"     • {memo['memo_title'][:60]}...")
            if len(memos) > 5:
                print(f"     ... and {len(memos) - 5} more")
        
        # Unmapped memos
        if unmapped_memos:
            print(f"\n❌ UNMAPPED MEMOS ({len(unmapped_memos)}):")
            for memo in unmapped_memos[:10]:
                title = memo.get('title', 'Untitled')
                filename = memo.get('filename', 'No filename')
                print(f"  • {title[:50]}... ({filename})")
            if len(unmapped_memos) > 10:
                print(f"  ... and {len(unmapped_memos) - 10} more")
        
        # Save to CSV
        self.save_to_csv()
        
        print("\n" + "=" * 80)
        print("✅ Report generated! See 'voice_memo_mappings.csv' for full details")
        print("=" * 80)
    
    def save_to_csv(self):
        """Save the mapping report to a CSV file"""
        csv_path = "voice_memo_mappings.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['memo_title', 'filename', 'cluster_id', 'book', 'part', 
                         'chapter', 'story_name', 'story_exists', 'memo_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for mapping in self.proposed_mappings:
                writer.writerow(mapping)
        
        logger.info(f"Saved detailed mappings to {csv_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze voice memo to story mappings')
    parser.add_argument('--voice-memos-db', required=True, help='Voice Memos database ID')
    parser.add_argument('--stories-db', required=True, help='Stories database ID')
    parser.add_argument('--token', help='Notion API token (or set NOTION_TOKEN env var)')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = VoiceMemoMappingAnalyzer(
            voice_memos_db_id=args.voice_memos_db,
            stories_db_id=args.stories_db,
            notion_token=args.token
        )
        
        # Analyze mappings
        analyzer.analyze_mappings()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())