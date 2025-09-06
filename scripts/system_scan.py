#!/usr/bin/env python3
"""
System-wide document scanner script for Local RAG Assistant.
This script can be run with sudo to scan the entire system.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.indexer import DocumentIndexer
from src.utils.logging import setup_logging


def check_sudo():
    """Check if running with sudo privileges."""
    return os.geteuid() == 0


def warn_system_scan():
    """Warn user about system scan implications."""
    print("=" * 60)
    print("SYSTEM-WIDE DOCUMENT SCAN WARNING")
    print("=" * 60)
    print()
    print("You are about to perform a system-wide document scan.")
    print("This operation will:")
    print()
    print("• Scan the entire filesystem for supported documents")
    print("• Access files from all users and system directories")
    print("• Index potentially sensitive documents")
    print("• Take a significant amount of time and resources")
    print("• Create a large knowledge base")
    print()
    print("Security considerations:")
    print("• Documents from /etc, /root, and other sensitive areas")
    print("• Personal files from all user directories")
    print("• System logs and configuration files")
    print()
    print("Privacy note:")
    print("• File paths can be hashed for privacy protection")
    print("• Content is indexed for search but not stored as-is")
    print("• Consider legal and privacy implications")
    print()
    print("=" * 60)


def get_user_confirmation():
    """Get user confirmation for system scan."""
    warn_system_scan()
    
    try:
        response = input("Do you want to continue with the system scan? (type 'yes' to confirm): ").strip().lower()
        return response == 'yes'
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="System-wide document scanner for Local RAG Assistant"
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Skip confirmation prompts (use with caution)'
    )
    
    parser.add_argument(
        '--max-documents', 
        type=int,
        help='Maximum number of documents to process'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be scanned without actually indexing'
    )
    
    args = parser.parse_args()
    
    # Check sudo
    if not check_sudo():
        print("Error: System scan requires sudo privileges.")
        print("Run with: sudo python scripts/system_scan.py")
        sys.exit(1)
    
    # Get confirmation unless forced
    if not args.force and not get_user_confirmation():
        print("System scan cancelled.")
        sys.exit(0)
    
    try:
        print("Initializing system scanner...")
        
        # Initialize indexer
        indexer = DocumentIndexer(args.config)
        
        if args.dry_run:
            print("DRY RUN: Scanning to show what would be processed...")
            
            # Just do the scan without indexing
            scan_result = indexer.document_loader.scan_documents('system')
            scan_summary = indexer.document_loader.get_scan_summary(scan_result)
            
            print("\nDry Run Results:")
            print(f"Total files found: {scan_summary['total_files_found']}")
            print(f"Documents to process: {scan_summary['total_documents']}")
            print(f"Total size: {scan_summary['total_size_formatted']}")
            print(f"File types: {scan_summary['file_types']}")
            print(f"Directories scanned: {scan_summary['directories_scanned']}")
            print(f"Permission errors: {scan_summary['permission_errors']}")
            print(f"Skipped files: {scan_summary['skipped_files']}")
            
            return
        
        # Perform actual indexing
        print("Starting system-wide document indexing...")
        start_time = time.time()
        
        result = indexer.scan_and_index(
            scan_mode='system',
            max_documents=args.max_documents
        )
        
        if result.get('cancelled'):
            print("Indexing was cancelled.")
            return
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("SYSTEM SCAN COMPLETED")
        print("=" * 60)
        print(f"Documents indexed: {result['indexed_documents']}")
        print(f"Total chunks created: {result['total_chunks']}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average processing speed: {result.get('documents_per_second', 0):.1f} docs/sec")
        
        scan_result = result['scan_result']
        print(f"\nFile Statistics:")
        print(f"Total files scanned: {scan_result['total_files_found']}")
        print(f"Total data size: {scan_result['total_size_formatted']}")
        print(f"File types processed: {len(scan_result['file_types'])}")
        print(f"Directories scanned: {scan_result['directories_scanned']}")
        
        if scan_result['permission_errors'] > 0:
            print(f"Permission errors: {scan_result['permission_errors']}")
        
        if scan_result['skipped_files'] > 0:
            print(f"Skipped files: {scan_result['skipped_files']}")
        
        print("\nThe Local RAG Assistant now has access to documents from across the system.")
        print("You can query this knowledge base using:")
        print("  make run      # CLI interface")
        print("  make web      # Web interface")
        print("  make api      # API server")
        
    except KeyboardInterrupt:
        print("\nSystem scan interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during system scan: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
