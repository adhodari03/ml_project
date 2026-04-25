import sys
import os
import subprocess
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_30c08a13e11a4f23b9e4996d73d25c6e.webm"
    result = subprocess.run([ffmpeg_exe, "-i", audio_path], stderr=subprocess.PIPE, text=True)
    for line in result.stderr.split('\n'):
        if "Duration" in line:
            print(line.strip())
except Exception as e:
    print(f"Error: {e}")
