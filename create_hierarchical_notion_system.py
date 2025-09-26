#!/usr/bin/env python3
"""
Create Hierarchical Notion System
Usage script to create and populate the four interconnected databases
"""

import os
import sys
import logging
from pathlib import Path
from hierarchical_notion_system import HierarchicalNotionSystem
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/hierarchical_system.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to create and populate the hierarchical system"""
    try:
        # Configuration
        NOTION_TOKEN = os.getenv('NOTION_TOKEN')
        PARENT_PAGE_ID = os.getenv('NOTION_PARENT_PAGE_ID')  # Parent page where databases will be created
        LIBRARY_PATH = "data/09 library full enhanced"  # Path to library data
        
        if not NOTION_TOKEN:
            logger.error("❌ NOTION_TOKEN environment variable not set")
            return
            
        if not PARENT_PAGE_ID:
            logger.error("❌ NOTION_PARENT_PAGE_ID environment variable not set")
            logger.info("💡 Set this to the ID of the parent page where you want the databases created")
            return
        
        # Check for command line flags
        fresh_start = "--fresh" in sys.argv
        
        # Check for threading configuration
        max_workers = 3  # Default
        for arg in sys.argv:
            if arg.startswith("--threads="):
                try:
                    max_workers = int(arg.split("=")[1])
                    logger.info(f"🧵 Using {max_workers} worker threads")
                except ValueError:
                    logger.warning(f"Invalid threads value: {arg}, using default {max_workers}")
        
        # Initialize progress tracker
        from progress_tracker import ProgressTracker
        progress_tracker = ProgressTracker()
        
        if fresh_start:
            logger.warning("⚠️ Fresh start requested - resetting all progress")
            progress_tracker.reset_progress()
        elif progress_tracker.has_existing_progress():
            logger.info("📂 Found existing progress - resuming from last checkpoint")
            progress_tracker.print_summary()
        
        logger.info("🚀 Starting Hierarchical Notion System Creation")
        logger.info(f"📁 Library Path: {LIBRARY_PATH}")
        logger.info(f"📄 Parent Page ID: {PARENT_PAGE_ID}")
        
        # Initialize the system with progress tracker and threading
        system = HierarchicalNotionSystem(
            notion_token=NOTION_TOKEN,
            parent_page_id=PARENT_PAGE_ID,
            progress_tracker=progress_tracker,
            max_workers=max_workers
        )
        
        # Step 1: Create all databases
        logger.info("🔨 Creating databases...")
        database_ids = system.create_all_databases()
        
        logger.info("✅ Database creation completed!")
        logger.info("🗄️ Created databases:")
        for name, db_id in database_ids.items():
            logger.info(f"  {name}: {db_id}")
        
        # Step 1.5: Rebuild progress tracker from existing Notion pages
        logger.info("🔄 Rebuilding progress tracker from existing pages...")
        system.rebuild_progress_tracker_from_notion()
        
        # Step 2: Load library data
        logger.info("📚 Loading library data...")
        library_data = system.load_library_data(LIBRARY_PATH)
        
        # Step 3: Populate databases
        logger.info("📝 Populating databases with library data...")
        created_pages = system.populate_databases(library_data)
        
        # Step 4: Create navigation guide
        logger.info("📋 Creating navigation guide...")
        guide_page_id = system.create_navigation_guide()
        
        # Step 5: Setup filtered views (manual instructions)
        logger.info("🔍 Setting up filtered views...")
        system.setup_filtered_views()
        
        # Step 6: Retry failed audio uploads
        logger.info("🔄 Retrying any failed audio uploads...")
        system.retry_failed_audio_uploads()
        
        # Step 7: Get performance statistics
        logger.info("📊 Gathering performance statistics...")
        perf_stats = system.get_performance_stats()
        
        # Print progress summary
        progress_tracker.print_summary()
        
        # Final summary
        logger.info("🎉 Hierarchical Notion System Creation Complete!")
        logger.info("📊 Final Summary:")
        logger.info(f"  🗄️ Databases Created: {len(database_ids)}")
        logger.info(f"  📚 Books: {len(created_pages['books'])} pages")
        logger.info(f"  📖 Chapters: {len(created_pages['chapters'])} pages")
        logger.info(f"  📝 Stories: {len(created_pages['stories'])} pages")
        logger.info(f"  🎙️ Voice Memos: {len(created_pages['voice_memos'])} pages")
        logger.info(f"  📋 Navigation Guide: {guide_page_id}")
        
        # Performance summary
        if perf_stats and "totals" in perf_stats:
            totals = perf_stats["totals"]
            logger.info(f"  🚀 API Calls Made: {totals['total_api_calls']}")
            logger.info(f"  💾 Cache Hit Rate: {totals['overall_cache_hit_rate_percent']}%")
            logger.info(f"  💰 Estimated Cost Savings: ${totals['estimated_total_cost_savings']:.2f}")
        
        # Print database URLs for easy access
        logger.info("\n🔗 Database URLs:")
        logger.info(f"  📚 Books: https://www.notion.so/{database_ids['books'].replace('-', '')}")
        logger.info(f"  📖 Chapters: https://www.notion.so/{database_ids['chapters'].replace('-', '')}")
        logger.info(f"  📝 Stories: https://www.notion.so/{database_ids['stories'].replace('-', '')}")
        logger.info(f"  🎙️ Voice Memos: https://www.notion.so/{database_ids['voice_memos'].replace('-', '')}")
        
        return database_ids, created_pages, guide_page_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create hierarchical system: {e}")
        raise

def create_sample_system():
    """Create a sample system with a subset of data for testing"""
    try:
        logger.info("🧪 Creating sample system for testing...")
        
        # Use environment variables or defaults
        NOTION_TOKEN = os.getenv('NOTION_TOKEN')
        PARENT_PAGE_ID = os.getenv('NOTION_PARENT_PAGE_ID')
        
        if not NOTION_TOKEN or not PARENT_PAGE_ID:
            logger.error("❌ Environment variables not set")
            return
        
        system = HierarchicalNotionSystem(
            notion_token=NOTION_TOKEN,
            parent_page_id=PARENT_PAGE_ID
        )
        
        # Create databases
        database_ids = system.create_all_databases()
        
        # Load only one book for testing
        library_data = system.load_library_data("data/09 library full enhanced")
        
        # Take only the first book
        sample_data = {k: v for k, v in list(library_data.items())[:1]}
        
        # Populate with sample data
        created_pages = system.populate_databases(sample_data)
        
        logger.info("✅ Sample system created successfully!")
        return database_ids, created_pages
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample system: {e}")
        raise

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        create_sample_system()
    else:
        main()