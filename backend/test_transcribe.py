import sys
import wave
import struct
import math
import traceback

sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")
from backend.models.speech import WhisperASR

# Generate a 1-second 440Hz sine wave audio file
def generate_tone(filename):
    sample_rate = 16000
    duration = 1.0
    freq = 440.0
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.cos(freq * math.pi * float(i) / float(sample_rate)))
            data = struct.pack('<h', value)
            f.writeframesraw(data)

audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/backend/test_tone.wav"
generate_tone(audio_path)

print("Starting transcription test...")
try:
    asr = WhisperASR()
    res = asr.transcribe(audio_path)
    print("Result:")
    print(res)
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
