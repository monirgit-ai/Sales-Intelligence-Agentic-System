"""
DeepSeek Monitoring Script
Continuously monitors DeepSeek API connectivity and functionality.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set DeepSeek API key
DEEPSEEK_KEY = ""
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_KEY

def check_imports() -> Dict[str, Any]:
    """Check if LangChain and required modules are available."""
    result = {
        "status": "unknown",
        "langchain_available": False,
        "langchain_version": None,
        "langchain_openai_available": False,
        "error": None
    }
    
    try:
        import langchain
        result["langchain_available"] = True
        result["langchain_version"] = langchain.__version__
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            result["langchain_openai_available"] = True
            result["status"] = "ok"
        except ImportError as e:
            result["error"] = f"LangChain OpenAI not available: {e}"
            result["status"] = "error"
            
    except ImportError as e:
        result["error"] = f"LangChain not available: {e}"
        result["status"] = "error"
    
    return result

def check_api_key() -> Dict[str, Any]:
    """Check if DeepSeek API key is configured."""
    result = {
        "status": "unknown",
        "key_found": False,
        "key_preview": None,
        "error": None
    }
    
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            result["key_found"] = True
            result["key_preview"] = f"{api_key[:10]}...{api_key[-5:]}" if len(api_key) > 15 else "***"
            result["status"] = "ok"
        else:
            result["error"] = "DEEPSEEK_API_KEY environment variable not set"
            result["status"] = "error"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    
    return result

def test_deepseek_connection() -> Dict[str, Any]:
    """Test actual connection to DeepSeek API."""
    result = {
        "status": "unknown",
        "connected": False,
        "response_time_ms": None,
        "response_text": None,
        "error": None
    }
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        
        api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            result["error"] = "DEEPSEEK_API_KEY not set"
            result["status"] = "error"
            return result
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            max_tokens=100,
            api_key=api_key,
            base_url=api_base
        )
        
        # Create simple prompt
        prompt = PromptTemplate(
            template="Say 'OK' if you can read this.",
            input_variables=[]
        )
        
        # Test connection
        start_time = time.time()
        chain = prompt | llm
        response = chain.invoke({})
        
        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)
        
        # Extract content
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        result["connected"] = True
        result["response_time_ms"] = response_time_ms
        result["response_text"] = response_text[:100]  # First 100 chars
        result["status"] = "ok"
        
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    
    return result

def test_coordinator_integration() -> Dict[str, Any]:
    """Test DeepSeek integration with Coordinator Agent."""
    result = {
        "status": "unknown",
        "integration_working": False,
        "narrative_generated": False,
        "narrative_length": None,
        "error": None
    }
    
    try:
        from src.agents.coordinator_agent import CoordinatorAgent
        
        logger.info("Testing Coordinator Agent integration...")
        coordinator = CoordinatorAgent(verbose=False)
        
        # Run comprehensive analysis (this will use DeepSeek)
        results = coordinator.run_all()
        
        if "narrative_report" in results:
            narrative = results["narrative_report"]
            result["narrative_generated"] = True
            result["narrative_length"] = len(narrative)
            result["integration_working"] = True
            result["status"] = "ok"
            
            # Check if it's LLM-generated (not fallback)
            if not narrative.startswith("Sales intelligence analysis completed"):
                result["deepseek_used"] = True
            else:
                result["deepseek_used"] = False
                result["status"] = "warning"
                result["error"] = "Using fallback synthesis - DeepSeek may not be active"
        else:
            result["error"] = "No narrative report generated"
            result["status"] = "error"
            
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    
    return result

def print_status_report():
    """Print comprehensive status report."""
    print("\n" + "="*70)
    print("DEEPSEEK MONITORING REPORT")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)
    
    # Check 1: Imports
    print("\n[1] Checking LangChain Availability...")
    import_status = check_imports()
    if import_status["status"] == "ok":
        print(f"   [OK] LangChain {import_status['langchain_version']} available")
        print(f"   [OK] LangChain OpenAI integration available")
    else:
        print(f"   [FAIL] {import_status['error']}")
        return
    
    # Check 2: API Key
    print("\n[2] Checking DeepSeek API Key...")
    key_status = check_api_key()
    if key_status["status"] == "ok":
        print(f"   [OK] API Key found: {key_status['key_preview']}")
    else:
        print(f"   [FAIL] {key_status['error']}")
        return
    
    # Check 3: API Connection
    print("\n[3] Testing DeepSeek API Connection...")
    connection_status = test_deepseek_connection()
    if connection_status["status"] == "ok":
        print(f"   [OK] Connected successfully!")
        print(f"   [INFO] Response time: {connection_status['response_time_ms']}ms")
        print(f"   [INFO] Response: {connection_status['response_text'][:50]}...")
    else:
        print(f"   [FAIL] Connection failed: {connection_status['error']}")
        return
    
    # Check 4: Coordinator Integration
    print("\n[4] Testing Coordinator Agent Integration...")
    integration_status = test_coordinator_integration()
    if integration_status["status"] == "ok":
        print(f"   [OK] Integration working!")
        print(f"   [OK] Narrative generated ({integration_status['narrative_length']} chars)")
        if integration_status.get("deepseek_used", False):
            print(f"   [OK] DeepSeek LLM actively generating narratives")
        else:
            print(f"   [WARN] Using fallback synthesis (check logs)")
    elif integration_status["status"] == "warning":
        print(f"   [WARN] {integration_status['error']}")
    else:
        print(f"   [FAIL] Integration failed: {integration_status['error']}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_ok = (
        import_status["status"] == "ok" and
        key_status["status"] == "ok" and
        connection_status["status"] == "ok" and
        integration_status["status"] == "ok"
    )
    
    if all_ok:
        print("[SUCCESS] DEEPSEEK IS FULLY OPERATIONAL")
        print("   - LangChain integration: Working")
        print("   - API key: Configured")
        print("   - API connection: Active")
        print("   - Coordinator integration: Functional")
        print("\n[SUCCESS] Your system is ready to use!")
    else:
        print("[WARNING] SOME ISSUES DETECTED")
        print("   Check the details above for specific problems.")
    
    print("="*70 + "\n")

def continuous_monitor(interval_seconds: int = 60):
    """Continuously monitor DeepSeek status."""
    print("\n" + "="*70)
    print("DEEPSEEK CONTINUOUS MONITORING")
    print("="*70)
    print(f"Monitoring every {interval_seconds} seconds...")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        while True:
            print_status_report()
            print(f"\n[INFO] Waiting {interval_seconds} seconds before next check...")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\n[STOP] Monitoring stopped by user")
        print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        continuous_monitor(interval)
    else:
        print_status_report()

