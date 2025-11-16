"""Quick test for DeepSeek integration - minimal dependencies"""
import os
import sys

# Set API key
os.environ["DEEPSEEK_API_KEY"] = "sk-64543b775f5546f0ba0e313366a1b550"

print("="*70)
print("QUICK DEEPSEEK + LANGCHAIN TEST")
print("="*70)
print(f"\nAPI Key: {os.getenv('DEEPSEEK_API_KEY')[:15]}...")

# Test LangChain import
print("\n[1] Testing LangChain import...")
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    try:
        from langchain_openai import ChatOpenAI
    except:
        from langchain.chat_models import ChatOpenAI
    print("    [OK] LangChain imported")
except ImportError as e:
    print(f"    [FAIL] {e}")
    print("\nInstall LangChain:")
    print("  pip install langchain openai")
    sys.exit(1)

# Test DeepSeek initialization
print("\n[2] Testing DeepSeek LLM initialization...")
try:
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.2,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com"
    )
    print("    [OK] DeepSeek LLM initialized")
except Exception as e:
    print(f"    [FAIL] {e}")
    sys.exit(1)

# Test simple call
print("\n[3] Testing DeepSeek API call...")
try:
    prompt = PromptTemplate(input_variables=["text"], template="Summarize: {text}")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run("Sales increased by 20%")
    print(f"    [OK] API call successful!")
    print(f"    Response: {result[:80]}...")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Coordinator Agent
print("\n[4] Testing with Coordinator Agent...")
try:
    from src.agents.coordinator_agent import CoordinatorAgent
    print("    [OK] CoordinatorAgent imported")
    print("    -> Running analysis (this may take 30-60 seconds)...")
    
    coordinator = CoordinatorAgent(verbose=True)
    results = coordinator.run_all()
    
    if "narrative_report" in results:
        narrative = results["narrative_report"]
        print(f"\n    [OK] Narrative generated ({len(narrative)} chars)")
        print("\n" + "-"*70)
        print("NARRATIVE (First 300 chars):")
        print("-"*70)
        print(narrative[:300] + "...")
        print("-"*70)
        
        if not narrative.startswith("Sales intelligence analysis completed"):
            print("\n    [SUCCESS] DeepSeek LLM generated the narrative!")
        else:
            print("\n    [WARNING] Using fallback - check logs above")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")

