import sys
import os
sys.path.insert(0, "/Users/aadeshdhodari/Downloads/ML_Project")

def get_audio_duration(file_path):
    try:
        import whisper.audio
        audio = whisper.audio.load_audio(file_path)
        duration = len(audio) / 16000.0
        print(f"Duration of {os.path.basename(file_path)}: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio: {e}")

get_audio_duration("/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_d167c9c6857344df8e5ede46c52e078b.webm")
get_audio_duration("/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_30c08a13e11a4f23b9e4996d73d25c6e.webm")
get_audio_duration("/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_fba00222999e4589a6f8952d1889a3cd.webm")
