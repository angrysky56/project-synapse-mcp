#!/usr/bin/env python3
"""
Minimal test to isolate the regex error source.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_step_by_step():
    """Test each component step by step to isolate the error."""
    print("🔍 Testing each component step by step...")
    
    try:
        # Test 1: Import semantic integrator
        print("Step 1: Importing semantic integrator...")
        from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
        print("✅ Import successful")
        
        # Test 2: Create instance
        print("Step 2: Creating instance...")
        integrator = SemanticIntegrator()
        print("✅ Instance created")
        
        # Test 3: Initialize (without Montague parser)
        print("Step 3: Initializing without parser...")
        await integrator.initialize()
        print("✅ Initialization successful")
        
        # Test 4: Test basic text processing only
        print("Step 4: Testing basic text processing...")
        result = await integrator._basic_text_processing("Hello world", "test", {})
        print("✅ Basic text processing successful")
        print(f"   Sentences: {len(result['sentences'])}")
        
        # Test 5: Test cleaning function directly
        print("Step 5: Testing cleaning function...")
        cleaned = await integrator._clean_text("Hello 'smart quotes' test")
        print(f"✅ Text cleaning successful: '{cleaned}'")
        
        # Test 6: Test full processing without Montague
        print("Step 6: Testing full processing without semantic analysis...")
        full_result = await integrator.process_text_with_semantics("Simple test", "test", {})
        print("✅ Full processing successful")
        print(f"   Entities: {len(full_result['entities'])}")
        print(f"   Facts: {len(full_result['facts'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error at current step: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the step-by-step test."""
    print("🧠 Project Synapse - Step-by-Step Debug")
    print("=" * 50)
    
    success = await test_step_by_step()
    
    if success:
        print("\n✅ All steps completed successfully!")
        print("The error must be in the knowledge graph connection or MCP layer.")
    else:
        print("\n❌ Error found in semantic processing layer.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
