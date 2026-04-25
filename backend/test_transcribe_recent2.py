import sys
sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.models.speech import WhisperASR

audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_30c08a13e11a4f23b9e4996d73d25c6e.webm"

print(f"Testing {audio_path}...")
asr = WhisperASR()
res = asr.transcribe(audio_path)
print("Result:")
print(res)
