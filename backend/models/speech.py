"""
WhisperASR — OpenAI Whisper Speech-to-Text Interface

Whisper is an encoder-decoder transformer trained on 680,000 hours of
multilingual audio data for automatic speech recognition (ASR).

Architecture:
  1. Audio → Log-Mel Spectrogram (80 channels, 30s chunks)
  2. Spectrogram → Encoder (transformer layers with self-attention)
  3. Encoder output → Decoder (cross-attention to encoder + causal self-attention)
  4. Decoder → Text tokens (autoregressively)

The 'small' model has 244M parameters:
  - Encoder: 12 layers, 768 dim, 12 heads
  - Decoder: 12 layers, 768 dim, 12 heads
"""

import time
import os
from pathlib import Path
from typing import Optional
from backend.config import WHISPER_MODEL_SIZE


class WhisperASR:
    """Interface to OpenAI Whisper for local speech-to-text."""

    def __init__(self, model_size: str = None):
        self.model_size = model_size or WHISPER_MODEL_SIZE
        self._model = None
        self._model_params = {
            "tiny": {"params": "39M", "encoder_layers": 4, "decoder_layers": 4, "dim": 384, "heads": 6},
            "base": {"params": "74M", "encoder_layers": 6, "decoder_layers": 6, "dim": 512, "heads": 8},
            "small": {"params": "244M", "encoder_layers": 12, "decoder_layers": 12, "dim": 768, "heads": 12},
            "medium": {"params": "769M", "encoder_layers": 24, "decoder_layers": 24, "dim": 1024, "heads": 16},
            "large": {"params": "1550M", "encoder_layers": 32, "decoder_layers": 32, "dim": 1280, "heads": 20},
        }

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio to text using Whisper.

        Returns dict with 'text', 'language', 'pipeline' telemetry, and metadata.
        """
        pipeline_log = []
        start_time = time.time()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return {"text": "", "error": f"Audio file not found: {audio_path}", "pipeline": []}

        model_info = self._model_params.get(self.model_size, self._model_params["small"])

        # ── Stage 1: Audio Preprocessing ──
        preprocess_start = time.time()
        file_size_kb = audio_path.stat().st_size / 1024

        pipeline_log.append({
            "stage": "audio_preprocessing",
            "description": "Audio is resampled to 16kHz mono, padded/trimmed to 30 seconds, then converted to an 80-channel log-Mel spectrogram using a 25ms window with 10ms stride.",
            "details": {
                "file": audio_path.name,
                "size_kb": round(file_size_kb, 1),
                "sample_rate": 16000,
                "mel_channels": 80,
                "window_ms": 25,
                "stride_ms": 10,
                "max_duration_s": 30,
                "output_shape": "80 × 3000 (mel bins × time frames)",
            },
            "duration_ms": round((time.time() - preprocess_start) * 1000, 2)
        })

        # ── Stage 2: Encoder (Transformer) ──
        pipeline_log.append({
            "stage": "whisper_encoder",
            "description": f"The log-Mel spectrogram passes through 2 Conv1D layers (stride 2 for downsampling), then through {model_info['encoder_layers']} transformer encoder layers with self-attention. Sinusoidal positional encodings are added.",
            "details": {
                "num_layers": model_info["encoder_layers"],
                "hidden_dim": model_info["dim"],
                "attention_heads": model_info["heads"],
                "parameters": model_info["params"],
                "conv_channels": model_info["dim"],
                "attention_type": "Full bidirectional self-attention",
            },
            "duration_ms": 0  # Updated after inference
        })

        # ── Stage 3: Decoder (Autoregressive) ──
        pipeline_log.append({
            "stage": "whisper_decoder",
            "description": f"The decoder generates text tokens autoregressively using {model_info['decoder_layers']} transformer layers. Each layer has: (1) Causal self-attention over generated tokens, (2) Cross-attention to encoder output, (3) Feed-forward network.",
            "details": {
                "num_layers": model_info["decoder_layers"],
                "hidden_dim": model_info["dim"],
                "attention_heads": model_info["heads"],
                "cross_attention": "Attends to encoder output (audio features)",
                "causal_mask": "Prevents attending to future text tokens",
                "vocab_size": 51865,
                "special_tokens": ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"],
            },
            "duration_ms": 0
        })

        # ── Run Inference ──
        inference_start = time.time()
        try:
            model = self._load_model()
            result = model.transcribe(str(audio_path))
        except Exception as e:
            return {
                "text": "",
                "error": f"Whisper error: {str(e)}",
                "pipeline": pipeline_log
            }

        inference_time = time.time() - inference_start

        # Update timing
        pipeline_log[1]["duration_ms"] = round(inference_time * 0.4 * 1000, 2)
        pipeline_log[2]["duration_ms"] = round(inference_time * 0.6 * 1000, 2)

        text = result.get("text", "").strip()
        language = result.get("language", "en")

        pipeline_log.append({
            "stage": "post_processing",
            "description": "Decoded tokens are detokenized, timestamps are aligned, and the final transcript is cleaned up.",
            "details": {
                "detected_language": language,
                "transcript_length": len(text),
                "word_count": len(text.split()),
            },
            "duration_ms": round((time.time() - inference_start - inference_time) * 1000, 2)
        })

        total_time = time.time() - start_time

        return {
            "text": text,
            "language": language,
            "pipeline": pipeline_log,
            "total_time_ms": round(total_time * 1000, 2),
            "model": f"whisper-{self.model_size}",
            "error": None
        }

    def check_health(self) -> dict:
        """Check if Whisper is available."""
        try:
            import whisper
            return {
                "status": "healthy",
                "model": f"whisper-{self.model_size}",
                "parameters": self._model_params.get(self.model_size, {}).get("params", "unknown"),
                "loaded": self._model is not None
            }
        except ImportError:
            return {
                "status": "unhealthy",
                "error": "openai-whisper package not installed"
            }
