import sys
import os
sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.orchestrator import Orchestrator

orch = Orchestrator()
try:
    orch.process_message("")
except Exception as e:
    import traceback
    traceback.print_exc()

