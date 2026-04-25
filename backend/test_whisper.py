import sys
import traceback
sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.models.speech import WhisperASR

try:
    print("Initializing WhisperASR...")
    asr = WhisperASR()
    print("Checking health...")
    print(asr.check_health())
    print("Loading model...")
    asr._load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")
    traceback.print_exc()
