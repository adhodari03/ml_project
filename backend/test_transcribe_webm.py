import sys
import traceback

sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.models.speech import WhisperASR

audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_fba00222999e4589a6f8952d1889a3cd.webm"

print(f"Testing {audio_path}...")
try:
    asr = WhisperASR()
    res = asr.transcribe(audio_path)
    print("Result:")
    print(res)
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
