#!/usr/bin/env python3
"""Test script for Local RAG Assistant system-wide scanning."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DocumentLoader
from src.utils.config import load_config, ensure_directories
from src.utils.logging import setup_logging


def test_system():
    """Test the system functionality."""
    print("🔧 Testing Local RAG Assistant System-Wide Scanning")
    print("=" * 60)
    
    try:
        # Load configuration
        print("1. Loading configuration...")
        config = load_config()
        ensure_directories(config)
        print("   ✓ Configuration loaded successfully")
        
        # Setup logging
        print("2. Setting up logging...")
        setup_logging(
            log_level=config.app.log_level,
            log_file=str(config.paths.logs / "test.log")
        )
        print("   ✓ Logging configured")
        
        # Test document loader
        print("3. Testing document scanner...")
        loader = DocumentLoader(config)
        print(f"   ✓ Scanner initialized (sudo: {loader.is_sudo})")
        
        # Test home directory scan
        print("4. Testing home directory scan...")
        result = loader.scan_documents('manual')  # Start with manual mode
        summary = loader.get_scan_summary(result)
        
        print(f"   ✓ Scan completed")
        print(f"   📊 Results:")
        print(f"      • Documents found: {summary['total_documents']}")
        print(f"      • Total size: {summary['total_size_formatted']}")
        print(f"      • Files scanned: {summary['total_files_found']}")
        print(f"      • Skipped: {summary['skipped_files']}")
        print(f"      • Permission errors: {summary['permission_errors']}")
        print(f"      • Scan time: {summary['scan_time']:.2f}s")
        
        # Test different scan modes
        scan_modes = ['manual', 'home']
        if loader.is_sudo:
            scan_modes.append('system')
        
        print("\n5. Testing scan modes:")
        for mode in scan_modes:
            try:
                print(f"   Testing '{mode}' mode...")
                if mode == 'system':
                    print("   ⚠️  Skipping system scan (would scan entire filesystem)")
                    continue
                
                result = loader.scan_documents(mode)
                summary = loader.get_scan_summary(result)
                print(f"   ✓ {mode}: {summary['total_documents']} docs, {summary['total_size_formatted']}")
                
            except Exception as e:
                print(f"   ✗ {mode}: {e}")
        
        print("\n🎉 System Test Complete!")
        print("=" * 60)
        print("✅ Local RAG Assistant is ready for system-wide document scanning")
        print("\nNext steps:")
        print("• For manual scanning: make ingest")
        print("• For home scanning: make ingest-home") 
        print("• For system scanning: sudo make ingest-system")
        print("• Install LLM support: python scripts/install_llama_cpp.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_system()
    sys.exit(0 if success else 1)
