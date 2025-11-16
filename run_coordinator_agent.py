#!/usr/bin/env python3
"""
Standalone script to run Coordinator Agent for n8n integration.
This script can be called directly from n8n Execute Command node.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.coordinator_agent import CoordinatorAgent
import logging

# Configure logging to output only to stderr (not stdout where JSON goes)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr
)

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-compatible types."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj

def main():
    """Main execution function."""
    try:
        # Initialize coordinator agent
        coordinator = CoordinatorAgent(verbose=True)
        
        # Run all analytics
        results = coordinator.run_all()
        
        # Extract key information for n8n
        prescriptive_actions = []
        if results.get('raw_results', {}).get('prescriptive', {}).get('recommendations'):
            prescriptive_actions = results['raw_results']['prescriptive']['recommendations']
        
        # Prepare output for n8n
        output = {
            'narrative_report': results.get('narrative_report', ''),
            'execution_summary': convert_to_serializable(results.get('execution_summary', {})),
            'summary_insights': convert_to_serializable(results.get('summary_insights', {})),
            'prescriptive_actions': convert_to_serializable(prescriptive_actions),
            'has_actions': len(prescriptive_actions) > 0,
            'status': 'success'
        }
        
        # Print JSON to stdout (n8n will capture this)
        print(json.dumps(output, indent=2))
        
        return 0
        
    except Exception as e:
        # Output error as JSON
        error_output = {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_output, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())

