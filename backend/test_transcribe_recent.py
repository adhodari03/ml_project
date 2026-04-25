import sys
sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.models.speech import WhisperASR

audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_d167c9c6857344df8e5ede46c52e078b.webm"

print(f"Testing {audio_path}...")
asr = WhisperASR()
res = asr.transcribe(audio_path)
print("Result:")
print(res)
