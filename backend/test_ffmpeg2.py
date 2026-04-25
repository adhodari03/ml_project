import sys
import os
import subprocess
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    audio_path = "/Users/aadeshdhodari/Downloads/ML_Project/data/uploads/audio_fba00222999e4589a6f8952d1889a3cd.webm"
    result = subprocess.run([ffmpeg_exe, "-i", audio_path], stderr=subprocess.PIPE, text=True)
    for line in result.stderr.split('\n'):
        if "Duration" in line:
            print(line.strip())
except Exception as e:
    print(f"Error: {e}")
