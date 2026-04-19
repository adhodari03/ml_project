"""
OllamaLLM — Llama 3 Interface via Ollama

This module wraps the Ollama Python client to provide a clean interface
for text generation using the Llama 3 model. It supports:
  - Standard text generation
  - Structured JSON output (for intent classification)
  - Streaming generation (for real-time UI updates)
  - Detailed pipeline telemetry (for the LLM visualizer)

The Llama 3 model is a decoder-only transformer that uses causal (masked)
self-attention for autoregressive text generation. Each token is generated
by attending only to previous tokens in the sequence.
"""

import json
import time
import ollama
from typing import Generator, Optional
from backend.config import OLLAMA_BASE_URL, LLM_MODEL


class OllamaLLM:
    """Interface to Llama 3 via Ollama for text generation and orchestration."""

    def __init__(self, model: str = None):
        self.model = model or LLM_MODEL
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self._pipeline_log = []

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7) -> dict:
        """
        Generate a response from Llama 3.

        Returns a dict with 'response' text and 'pipeline' telemetry data
        showing each stage of the transformer inference pipeline.
        """
        self._pipeline_log = []
        start_time = time.time()

        # ── Stage 1: Tokenization ──
        token_start = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Approximate token count (rough estimate for visualization)
        approx_tokens = len(prompt.split()) + (len(system.split()) if system else 0)
        self._pipeline_log.append({
            "stage": "tokenization",
            "description": "Input text is split into tokens (subword units) and converted to integer IDs. Each token maps to a learned embedding vector.",
            "details": {
                "input_length_chars": len(prompt),
                "approx_tokens": approx_tokens,
                "model": self.model,
            },
            "duration_ms": round((time.time() - token_start) * 1000, 2)
        })

        # ── Stage 2: Embedding Lookup ──
        embed_start = time.time()
        self._pipeline_log.append({
            "stage": "embedding",
            "description": "Each token ID is mapped to a dense vector (dimension 4096 for Llama 3 8B). Positional encodings (RoPE) are added to inject sequence order information.",
            "details": {
                "embedding_dim": 4096,
                "positional_encoding": "RoPE (Rotary Position Embedding)",
                "vocab_size": 128256,
            },
            "duration_ms": round((time.time() - embed_start) * 1000, 2)
        })

        # ── Stage 3: Transformer Forward Pass ──
        inference_start = time.time()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
        except Exception as e:
            return {
                "response": f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running with `ollama serve`.",
                "pipeline": self._pipeline_log,
                "error": True
            }

        inference_time = time.time() - inference_start

        self._pipeline_log.append({
            "stage": "transformer_layers",
            "description": "Input passes through 32 transformer decoder layers. Each layer applies: (1) Multi-Head Self-Attention with causal mask, (2) RMSNorm, (3) Feed-Forward Network (SwiGLU activation), (4) Residual connections.",
            "details": {
                "num_layers": 32,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_type": "Grouped-Query Attention (GQA)",
                "ffn_hidden_dim": 14336,
                "activation": "SwiGLU",
                "normalization": "RMSNorm (pre-norm)",
                "context_window": 8192,
            },
            "duration_ms": round(inference_time * 1000, 2)
        })

        # ── Stage 4: Output Generation ──
        output_text = response["message"]["content"]
        output_tokens = len(output_text.split())

        self._pipeline_log.append({
            "stage": "output_generation",
            "description": "The final hidden states are projected through a linear layer to logits over the vocabulary. Softmax converts logits to probabilities. Tokens are sampled autoregressively until EOS.",
            "details": {
                "output_tokens": output_tokens,
                "temperature": temperature,
                "sampling": "multinomial" if temperature > 0 else "greedy",
                "tokens_per_second": round(output_tokens / max(inference_time, 0.001), 1),
            },
            "duration_ms": round((time.time() - start_time) * 1000, 2)
        })

        total_time = time.time() - start_time

        return {
            "response": output_text,
            "pipeline": self._pipeline_log,
            "total_time_ms": round(total_time * 1000, 2),
            "model": self.model,
            "error": False
        }

    def generate_structured(self, prompt: str, system: str = None) -> dict:
        """
        Generate a structured JSON response from Llama 3.
        Used for intent classification in the orchestrator.
        """
        result = self.generate(prompt, system=system, temperature=0.1)

        if result.get("error"):
            return result

        # Parse JSON from response
        try:
            text = result["response"]
            # Try to extract JSON from the response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = json.loads(text.strip())
            result["parsed"] = parsed
        except (json.JSONDecodeError, IndexError):
            result["parsed"] = {
                "intent": "general_chat",
                "confidence": 0.5,
                "reasoning": "Failed to parse structured output, defaulting to general chat",
                "extracted_entities": {}
            }

        return result

    def stream(self, prompt: str, system: str = None, temperature: float = 0.7) -> Generator[dict, None, None]:
        """
        Stream response tokens from Llama 3 for real-time UI updates.
        Yields dicts with 'token' and optional 'done' fields.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature}
            )

            for chunk in stream:
                yield {
                    "token": chunk["message"]["content"],
                    "done": chunk.get("done", False)
                }
        except Exception as e:
            yield {
                "token": f"Error: {str(e)}",
                "done": True,
                "error": True
            }

    def check_health(self) -> dict:
        """Check if the Ollama server and Llama 3 model are available."""
        try:
            models = self.client.list()
            available = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
            model_found = any(self.model in name for name in available)
            return {
                "status": "healthy" if model_found else "model_not_found",
                "model": self.model,
                "available_models": available,
                "ollama_running": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model,
                "error": str(e),
                "ollama_running": False
            }
