#!/usr/bin/env python3
"""
Match Voice Memos to Existing Stories in Notion

This script:
1. Loads the voice memo mappings from the CSV
2. Queries existing stories in Notion
3. Finds the best match for each cluster
4. Updates voice memos to link to the correct existing stories
"""

import os
import csv
import json
import logging
from typing import Dict, List, Optional, Tuple
from notion_service import NotionService
from difflib import SequenceMatcher
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceMemoStoryMatcher:
    """Matches voice memos to existing stories in Notion"""
    
    def __init__(self, 
                 voice_memos_db_id: str,
                 stories_db_id: str,
                 notion_token: str = None,
                 mappings_csv: str = "voice_memo_mappings.csv",
                 cluster_descriptions_file: str = "data/06 cluster_descriptions/medium.json"):
        """Initialize the matcher"""
        self.notion_token = notion_token or os.getenv('NOTION_TOKEN')
        if not self.notion_token:
            raise ValueError("NOTION_TOKEN not provided and not found in environment")
        
        self.voice_memos_db_id = voice_memos_db_id
        self.stories_db_id = stories_db_id
        self.mappings_csv = mappings_csv
        
        # Initialize Notion services
        self.voice_memos_service = NotionService(voice_memos_db_id, self.notion_token)
        self.stories_service = NotionService(stories_db_id, self.notion_token)
        
        # Load cluster descriptions
        self.cluster_descriptions = self._load_cluster_descriptions(cluster_descriptions_file)
        
        # Will be populated
        self.existing_stories = {}
        self.cluster_to_story_map = {}
        self.voice_memo_mappings = []
        
        # Statistics
        self.stats = {
            'total_memos': 0,
            'matched_clusters': 0,
            'unmatched_clusters': 0,
            'successful_assignments': 0,
            'failed_assignments': 0
        }
    
    def _load_cluster_descriptions(self, cluster_descriptions_file: str) -> Dict[int, str]:
        """Load cluster descriptions"""
        descriptions = {}
        try:
            with open(cluster_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                clusters = data.get('clusters', [])
                
                for cluster in clusters:
                    cluster_id = int(cluster.get('cluster_id', 0))
                    title = cluster.get('title', '')
                    if cluster_id and title:
                        descriptions[cluster_id] = title
                
                logger.info(f"✅ Loaded {len(descriptions)} cluster descriptions")
        except Exception as e:
            logger.warning(f"⚠️ Could not load cluster descriptions: {e}")
        
        return descriptions
    
    def load_voice_memo_mappings(self):
        """Load the voice memo mappings from CSV"""
        try:
            with open(self.mappings_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                self.voice_memo_mappings = list(reader)
                self.stats['total_memos'] = len(self.voice_memo_mappings)
                logger.info(f"✅ Loaded {len(self.voice_memo_mappings)} voice memo mappings")
                
                # Get unique clusters
                unique_clusters = set()
                for mapping in self.voice_memo_mappings:
                    cluster_id = mapping.get('cluster_id')
                    if cluster_id:
                        unique_clusters.add(int(cluster_id))
                
                logger.info(f"📊 Found {len(unique_clusters)} unique clusters to match")
                return unique_clusters
                
        except Exception as e:
            logger.error(f"❌ Error loading CSV mappings: {e}")
            return set()
    
    def query_existing_stories(self) -> Dict[str, Dict]:
        """Query all existing stories from Notion"""
        logger.info("🔍 Querying existing stories in Notion...")
        
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
                        self.existing_stories[story_name] = {
                            'id': story_id,
                            'name': story_name,
                            'properties': properties
                        }
            
            logger.info(f"✅ Found {len(self.existing_stories)} existing stories")
            return self.existing_stories
            
        except Exception as e:
            logger.error(f"❌ Error querying stories: {e}")
            return {}
    
    def find_best_story_match(self, cluster_id: int, cluster_description: str) -> Tuple[Optional[str], Optional[str], float]:
        """Find the best matching existing story for a cluster"""
        best_match = None
        best_score = 0
        best_story_id = None
        
        # Try different matching strategies
        search_terms = []
        
        # 1. Try the cluster description directly
        if cluster_description:
            search_terms.append(cluster_description)
        
        # 2. Try from loaded descriptions
        if cluster_id in self.cluster_descriptions:
            search_terms.append(self.cluster_descriptions[cluster_id])
        
        # 3. Try with "Story X:" prefix
        search_terms.append(f"Story {cluster_id}:")
        
        for search_term in search_terms:
            for story_name, story_data in self.existing_stories.items():
                # Calculate similarity score
                score = SequenceMatcher(None, search_term.lower(), story_name.lower()).ratio()
                
                # Also check if cluster ID is in the story name
                if f"{cluster_id}" in story_name:
                    score += 0.3  # Boost score if cluster ID matches
                
                # Check for key words match
                search_words = set(search_term.lower().split())
                story_words = set(story_name.lower().split())
                common_words = search_words.intersection(story_words)
                if len(common_words) > 2:  # At least 3 common words
                    score += 0.2 * (len(common_words) / len(search_words))
                
                if score > best_score:
                    best_score = score
                    best_match = story_name
                    best_story_id = story_data['id']
        
        return best_match, best_story_id, best_score
    
    def match_clusters_to_stories(self, unique_clusters: set):
        """Match each cluster to an existing story"""
        logger.info("\n" + "=" * 60)
        logger.info("MATCHING CLUSTERS TO EXISTING STORIES")
        logger.info("=" * 60)
        
        for cluster_id in sorted(unique_clusters):
            # Get cluster description from mappings or descriptions
            cluster_desc = self.cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
            
            # Find best match
            story_name, story_id, score = self.find_best_story_match(cluster_id, cluster_desc)
            
            if story_id and score > 0.3:  # Threshold for acceptable match
                self.cluster_to_story_map[cluster_id] = {
                    'story_id': story_id,
                    'story_name': story_name,
                    'match_score': score
                }
                self.stats['matched_clusters'] += 1
                
                status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❓"
                logger.info(f"{status} Cluster {cluster_id}: {cluster_desc[:50]}...")
                logger.info(f"   → {story_name} (score: {score:.2f})")
            else:
                self.stats['unmatched_clusters'] += 1
                logger.warning(f"❌ Cluster {cluster_id}: {cluster_desc[:50]}... - NO MATCH FOUND")
        
        logger.info(f"\n📊 Matched {self.stats['matched_clusters']}/{len(unique_clusters)} clusters")
    
    def update_voice_memos(self, dry_run: bool = False):
        """Update voice memos with their story relations"""
        logger.info("\n" + "=" * 60)
        logger.info("UPDATING VOICE MEMOS" + (" (DRY RUN)" if dry_run else ""))
        logger.info("=" * 60)
        
        updates_by_story = {}
        
        # Group updates by story
        for mapping in self.voice_memo_mappings:
            cluster_id = int(mapping.get('cluster_id', 0))
            memo_id = mapping.get('memo_id')
            memo_title = mapping.get('memo_title', 'Untitled')
            
            if cluster_id in self.cluster_to_story_map:
                story_info = self.cluster_to_story_map[cluster_id]
                story_name = story_info['story_name']
                story_id = story_info['story_id']
                
                if story_name not in updates_by_story:
                    updates_by_story[story_name] = []
                
                updates_by_story[story_name].append({
                    'memo_id': memo_id,
                    'memo_title': memo_title,
                    'story_id': story_id
                })
        
        # Process updates
        for story_name, memos in updates_by_story.items():
            logger.info(f"\n📚 {story_name} ({len(memos)} memos)")
            
            for i, memo in enumerate(memos[:5]):  # Show first 5
                logger.info(f"   • {memo['memo_title'][:60]}...")
            
            if len(memos) > 5:
                logger.info(f"   ... and {len(memos) - 5} more")
            
            if not dry_run:
                # Actually update the memos
                for memo in memos:
                    try:
                        properties = {
                            "Story": {
                                "relation": [{"id": memo['story_id']}]
                            }
                        }
                        
                        success = self.voice_memos_service.update_page(memo['memo_id'], properties)
                        
                        if success:
                            self.stats['successful_assignments'] += 1
                        else:
                            self.stats['failed_assignments'] += 1
                            logger.error(f"   ❌ Failed to update: {memo['memo_title'][:40]}...")
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        self.stats['failed_assignments'] += 1
                        logger.error(f"   ❌ Error updating {memo['memo_title'][:40]}...: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Total voice memos: {self.stats['total_memos']}")
        logger.info(f"✅ Matched clusters: {self.stats['matched_clusters']}")
        logger.info(f"❌ Unmatched clusters: {self.stats['unmatched_clusters']}")
        
        if not dry_run:
            logger.info(f"✅ Successful assignments: {self.stats['successful_assignments']}")
            logger.info(f"❌ Failed assignments: {self.stats['failed_assignments']}")
        else:
            logger.info(f"🔍 DRY RUN - No changes made")
    
    def run(self, dry_run: bool = False):
        """Run the complete matching and assignment process"""
        # Load mappings
        unique_clusters = self.load_voice_memo_mappings()
        if not unique_clusters:
            logger.error("No mappings loaded, exiting")
            return
        
        # Query existing stories
        self.query_existing_stories()
        if not self.existing_stories:
            logger.error("No existing stories found, exiting")
            return
        
        # Match clusters to stories
        self.match_clusters_to_stories(unique_clusters)
        
        # Update voice memos
        self.update_voice_memos(dry_run=dry_run)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Match and assign voice memos to existing stories')
    parser.add_argument('--voice-memos-db', required=True, help='Voice Memos database ID')
    parser.add_argument('--stories-db', required=True, help='Stories database ID')
    parser.add_argument('--token', help='Notion API token (or set NOTION_TOKEN env var)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--csv', default='voice_memo_mappings.csv', help='Path to mappings CSV file')
    
    args = parser.parse_args()
    
    try:
        # Create matcher
        matcher = VoiceMemoStoryMatcher(
            voice_memos_db_id=args.voice_memos_db,
            stories_db_id=args.stories_db,
            notion_token=args.token,
            mappings_csv=args.csv
        )
        
        # Run the matching process
        matcher.run(dry_run=args.dry_run)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())