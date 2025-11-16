#!/usr/bin/env python3
"""
Run Coordinator Agent with monitoring for LangChain + DeepSeek integration
This script will show detailed logs of the entire process.
"""

import os
import sys
import logging
from pathlib import Path

# Set DeepSeek API key
os.environ["DEEPSEEK_API_KEY"] = "sk-64543b775f5546f0ba0e313366a1b550"

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

print("="*80)
print("SALES INTELLIGENCE AGENTIC SYSTEM - MONITORED EXECUTION")
print("="*80)
print(f"\nDeepSeek API Key: {os.getenv('DEEPSEEK_API_KEY')[:15]}...")
print("\n" + "-"*80)
print("WATCHING FOR:")
print("  [1] LangChain initialization")
print("  [2] DeepSeek LLM setup")
print("  [3] Analytics execution")
print("  [4] DeepSeek API calls")
print("  [5] Narrative generation")
print("-"*80 + "\n")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("[STEP 1] Importing Coordinator Agent...")
    from src.agents.coordinator_agent import CoordinatorAgent
    print("   [OK] CoordinatorAgent imported successfully\n")
    
    print("[STEP 2] Initializing Coordinator Agent...")
    coordinator = CoordinatorAgent(verbose=True)
    print("   [OK] CoordinatorAgent initialized\n")
    
    print("[STEP 3] Running comprehensive analysis...")
    print("   This will execute:")
    print("     - Descriptive analytics")
    print("     - Diagnostic analytics")
    print("     - Predictive analytics")
    print("     - Prescriptive analytics")
    print("     - LangChain + DeepSeek narrative synthesis")
    print("\n   Watch for 'Initialized DeepSeek LLM' in logs below...\n")
    print("="*80)
    print("EXECUTION LOGS:")
    print("="*80 + "\n")
    
    results = coordinator.run_all()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    if "narrative_report" in results:
        narrative = results["narrative_report"]
        print(f"\nNarrative Report Length: {len(narrative)} characters")
        print("\n" + "-"*80)
        print("NARRATIVE REPORT (First 500 characters):")
        print("-"*80)
        print(narrative[:500])
        if len(narrative) > 500:
            print("...")
        print("-"*80)
        
        # Check if DeepSeek was used
        if not narrative.startswith("Sales intelligence analysis completed"):
            print("\n[SUCCESS] DeepSeek LLM generated the narrative!")
            print("   LangChain + DeepSeek integration is WORKING!")
        else:
            print("\n[WARNING] Using fallback synthesis")
            print("   Check logs above for DeepSeek initialization errors")
    
    if "execution_summary" in results:
        print("\n" + "-"*80)
        print("EXECUTION SUMMARY:")
        print("-"*80)
        for stage, status in results["execution_summary"].get("stages", {}).items():
            icon = "[OK]" if status == "success" else "[FAIL]"
            print(f"   {icon} {stage}: {status}")
    
    print("\n" + "="*80)
    print("MONITORING COMPLETE")
    print("="*80)
    print("\nTo see if DeepSeek was used, check the logs above for:")
    print("  - 'Initialized DeepSeek LLM: deepseek-chat'")
    print("  - 'Making LLM call via LangChain LLMChain...'")
    print("  - 'Successfully generated narrative using LangChain LLM'\n")

except Exception as e:
    print(f"\n[ERROR] Execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

