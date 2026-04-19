"""
NexusAI — FastAPI Server

Main entry point for the NexusAI backend. Provides REST endpoints
for all input modalities (text, voice, image, document) and a
WebSocket endpoint for streaming responses.
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import time
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    SERVER_HOST, SERVER_PORT, CORS_ORIGINS,
    UPLOAD_DIR, FRONTEND_DIR
)
from backend.orchestrator import Orchestrator


# ─── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="NexusAI",
    description="Multimodal Agentic Assistant with LLM Orchestration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator (lazy — created on first request)
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# ─── Request/Response Models ───────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    agent_override: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    agent: str
    intent: str
    confidence: float
    reasoning: str
    pipeline: list
    total_time_ms: float
    memory_stats: dict


# ─── API Endpoints ──────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Check system health — Ollama, models, and memory."""
    try:
        orch = get_orchestrator()
        health = orch.get_health()
        return {"status": "ok", "components": health}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/memory/clear")
async def clear_memory():
    """Clear conversation history from ChromaDB."""
    try:
        orch = get_orchestrator()
        success = orch.memory.clear_conversations()
        return {"status": "ok", "cleared": success}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process a text message through the orchestration pipeline.

    Flow: Message → Intent Classification → RAG Retrieval → Agent → Response
    """
    try:
        orch = get_orchestrator()
        result = orch.process_message(request.message, agent_override=request.agent_override)
        return result
    except Exception as e:
        return {
            "response": f"Error processing message: {str(e)}",
            "agent": "System",
            "intent": "error",
            "confidence": 0,
            "reasoning": str(e),
            "pipeline": [],
            "total_time_ms": 0,
            "memory_stats": {}
        }


@app.post("/api/voice")
async def voice_input(audio: UploadFile = File(...)):
    """
    Process voice input: Audio → Whisper (ASR) → Text → Orchestrator.

    Pipeline visualization shows both the Whisper encoder-decoder pipeline
    and the subsequent LLM orchestration pipeline.
    """
    try:
        # Save uploaded audio
        audio_path = UPLOAD_DIR / f"audio_{uuid.uuid4().hex}{Path(audio.filename).suffix}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Transcribe with Whisper
        from backend.models.speech import WhisperASR
        whisper = WhisperASR()
        transcription = whisper.transcribe(str(audio_path))

        if transcription.get("error"):
            return {
                "response": f"Transcription error: {transcription['error']}",
                "transcription": "",
                "pipeline": transcription.get("pipeline", []),
                "agent": "Whisper",
                "intent": "error",
                "confidence": 0,
                "reasoning": "",
                "total_time_ms": 0,
                "memory_stats": {}
            }

        text = transcription["text"]
        whisper_pipeline = transcription.get("pipeline", [])

        # Process transcribed text through orchestrator
        orch = get_orchestrator()
        result = orch.process_message(text)

        # Combine pipelines
        combined_pipeline = whisper_pipeline + result.get("pipeline", [])

        return {
            **result,
            "transcription": text,
            "pipeline": combined_pipeline,
        }
    except Exception as e:
        return {
            "response": f"Voice processing error: {str(e)}",
            "transcription": "",
            "pipeline": [],
            "agent": "System",
            "intent": "error",
            "confidence": 0,
            "reasoning": str(e),
            "total_time_ms": 0,
            "memory_stats": {}
        }


@app.post("/api/image")
async def image_input(
    image: UploadFile = File(...),
    message: str = Form(default="Describe this image in detail.")
):
    """
    Process image input: Image → LLaVA → Analysis + Orchestrator.
    """
    try:
        # Save uploaded image
        image_path = UPLOAD_DIR / f"img_{uuid.uuid4().hex}{Path(image.filename).suffix}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Process through orchestrator with image metadata
        orch = get_orchestrator()
        result = orch.process_message(
            message,
            metadata={"image_path": str(image_path)}
        )

        return result
    except Exception as e:
        return {
            "response": f"Image processing error: {str(e)}",
            "pipeline": [],
            "agent": "System",
            "intent": "error",
            "confidence": 0,
            "reasoning": str(e),
            "total_time_ms": 0,
            "memory_stats": {}
        }


@app.post("/api/document")
async def upload_document(document: UploadFile = File(...)):
    """
    Upload and ingest a document into the RAG memory store.

    Pipeline: Document → Text Extraction → Chunking → Embedding → ChromaDB
    """
    try:
        # Save uploaded document
        doc_path = UPLOAD_DIR / f"doc_{uuid.uuid4().hex}{Path(document.filename).suffix}"
        with open(doc_path, "wb") as f:
            shutil.copyfileobj(document.file, f)

        # Ingest into RAG store
        orch = get_orchestrator()
        doc_agent = orch.agents.get("document_qa")
        if doc_agent:
            result = doc_agent.ingest_document(str(doc_path))
            return {
                "status": "success",
                "filename": document.filename,
                **result
            }
        else:
            return {"status": "error", "error": "Document Q&A agent not available"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/history")
async def get_history():
    """Get conversation history from ChromaDB."""
    try:
        orch = get_orchestrator()
        # Get recent conversations
        collection = orch.memory.conversations
        if collection.count() == 0:
            return {"messages": []}

        results = collection.get(
            limit=50,
            include=["documents", "metadatas"]
        )

        messages = []
        if results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                messages.append({
                    "text": doc,
                    "role": meta.get("role", "unknown"),
                    "timestamp": meta.get("timestamp", ""),
                    "agent": meta.get("agent", ""),
                    "intent": meta.get("intent", ""),
                })

        # Sort by timestamp
        messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return {"messages": messages[:50]}
    except Exception as e:
        return {"messages": [], "error": str(e)}


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks from the task manager."""
    try:
        orch = get_orchestrator()
        task_agent = orch.agents.get("task_management")
        if task_agent:
            return {"tasks": task_agent.get_all_tasks()}
        return {"tasks": []}
    except Exception as e:
        return {"tasks": [], "error": str(e)}


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory store statistics."""
    try:
        orch = get_orchestrator()
        return orch.memory.get_stats()
    except Exception as e:
        return {"error": str(e)}


# ─── Tier 2: Embedding Space Visualization ──────────────────

@app.get("/api/embeddings/visualize")
async def visualize_embeddings():
    """
    Project all stored embeddings to 2D using t-SNE.

    Returns points with (x, y) coordinates, labels, and metadata
    for interactive scatter plot visualization.
    """
    try:
        orch = get_orchestrator()
        all_points = []
        colors_map = {
            "conversations": {"color": "#818cf8", "label": "Conversations"},
            "documents": {"color": "#34d399", "label": "Documents"},
            "tasks": {"color": "#60a5fa", "label": "Tasks"},
        }

        collections = {
            "conversations": orch.memory.conversations,
            "documents": orch.memory.documents,
            "tasks": orch.memory.tasks_collection,
        }

        embeddings = []
        labels = []
        texts = []
        categories = []

        for name, col in collections.items():
            count = col.count()
            if count == 0:
                continue
            data = col.get(
                limit=min(count, 100),
                include=["embeddings", "documents", "metadatas"]
            )
            if data["embeddings"]:
                for emb, doc, meta in zip(data["embeddings"], data["documents"], data["metadatas"]):
                    embeddings.append(emb)
                    labels.append(name)
                    texts.append(doc[:80] if doc else "")
                    categories.append(colors_map[name])

        if len(embeddings) < 3:
            return {
                "points": [],
                "message": f"Need at least 3 vectors for t-SNE projection, currently have {len(embeddings)}. Send more messages or upload a document.",
                "total_vectors": len(embeddings)
            }

        import numpy as np
        from sklearn.manifold import TSNE

        X = np.array(embeddings)
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
        X_2d = tsne.fit_transform(X)

        # Normalize to [-1, 1]
        X_2d -= X_2d.mean(axis=0)
        scale = np.abs(X_2d).max()
        if scale > 0:
            X_2d /= scale

        points = []
        for i, (x, y) in enumerate(X_2d):
            points.append({
                "x": float(x),
                "y": float(y),
                "text": texts[i],
                "category": labels[i],
                "color": categories[i]["color"],
            })

        return {
            "points": points,
            "total_vectors": len(embeddings),
            "dimensions_original": len(embeddings[0]),
            "algorithm": "t-SNE",
            "perplexity": perplexity,
            "categories": list(colors_map.values()),
        }
    except Exception as e:
        return {"points": [], "error": str(e)}


# ─── Tier 2: Prompt Engineering Lab ─────────────────────────

@app.post("/api/prompt-lab")
async def prompt_lab(request: ChatRequest):
    """
    Show the complete assembled prompt — system prompt, RAG context,
    and user message — as it's sent to the LLM.
    """
    try:
        orch = get_orchestrator()

        # Step 1: Get intent classification prompt
        from backend.config import ORCHESTRATOR_SYSTEM_PROMPT, GENERAL_CHAT_PROMPT

        classification = orch.llm.generate_structured(
            prompt=request.message,
            system=ORCHESTRATOR_SYSTEM_PROMPT
        )
        intent_data = classification.get("parsed", {})
        intent = intent_data.get("intent", "general_chat")

        # Step 2: Get RAG context
        retrieval = orch.memory.retrieve_context(request.message)
        context = retrieval.get("results", [])

        context_str = ""
        if context:
            context_str = "### Conversation History:\n"
            for item in context:
                role = item.get("metadata", {}).get("role", "unknown")
                text = item.get("text", "")
                context_str += f"[{role}]: {text}\n"
            context_str += "### End of History\n\n"

        # Step 3: Build agent prompt
        agent_system = GENERAL_CHAT_PROMPT
        if intent in orch.agents:
            agent = orch.agents[intent]
            agent_system = getattr(agent, 'system_prompt', GENERAL_CHAT_PROMPT)

        assembled_prompt = f"{context_str}User: {request.message}"

        return {
            "stages": [
                {
                    "name": "1. Intent Classification",
                    "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
                    "user_input": request.message,
                    "result": intent_data,
                    "description": "The orchestrator's system prompt instructs Llama 3 to output structured JSON. This works because Llama 3 is instruction-tuned (SFT + RLHF) — it learned to follow output format instructions during fine-tuning.",
                },
                {
                    "name": "2. RAG Context Retrieved",
                    "context_items": len(context),
                    "context_text": context_str or "(no matching context found)",
                    "top_similarity": context[0]["similarity"] if context else None,
                    "description": "Retrieved context is prepended to the user's message, giving the LLM access to relevant past conversations or document passages.",
                },
                {
                    "name": "3. Assembled Prompt → Agent",
                    "system_prompt": agent_system,
                    "assembled_user_prompt": assembled_prompt,
                    "selected_agent": intent,
                    "description": "The final prompt sent to the LLM is: [system prompt] + [RAG context] + [user message]. The system prompt defines the agent's personality and capabilities.",
                },
            ],
            "intent": intent,
            "total_prompt_chars": len(agent_system) + len(assembled_prompt),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Tier 2: Model Comparison ───────────────────────────────

@app.post("/api/models/compare")
async def compare_models(request: ChatRequest):
    """
    Run the same prompt through all available Ollama models
    and compare response quality, latency, and token usage.
    """
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")

        # Get available models
        models_list = client.list()
        chat_models = [
            m.model for m in models_list.models
            if 'embed' not in m.model.lower()
        ]

        if not chat_models:
            return {"results": [], "error": "No chat models found"}

        results = []
        for model_name in chat_models:
            start = time.time()
            try:
                resp = client.generate(
                    model=model_name,
                    prompt=request.message,
                    options={"temperature": 0.7, "num_predict": 150}
                )
                elapsed = time.time() - start
                response_text = resp.get("response", "")
                eval_count = resp.get("eval_count", 0)
                tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0

                results.append({
                    "model": model_name,
                    "response": response_text,
                    "latency_ms": round(elapsed * 1000),
                    "tokens_generated": eval_count,
                    "tokens_per_second": round(tokens_per_sec, 1),
                    "response_length": len(response_text),
                    "status": "success",
                })
            except Exception as e:
                results.append({
                    "model": model_name,
                    "response": "",
                    "latency_ms": round((time.time() - start) * 1000),
                    "tokens_generated": 0,
                    "tokens_per_second": 0,
                    "response_length": 0,
                    "status": f"error: {str(e)}",
                })

        return {
            "prompt": request.message,
            "results": results,
            "models_tested": len(results),
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


# ─── WebSocket for Streaming ─────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")

            if not message:
                continue

            # Process through orchestrator
            orch = get_orchestrator()
            result = orch.process_message(message)

            # Send full result
            await websocket.send_text(json.dumps(result, default=str))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass


# ─── Static Files (Frontend) ────────────────────────────────

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend dashboard."""
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>NexusAI</h1><p>Frontend not found.</p>")


# ─── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NexusAI Server...")
    print(f"   Dashboard: http://localhost:{SERVER_PORT}")
    print(f"   API Docs:  http://localhost:{SERVER_PORT}/docs")
    uvicorn.run(
        "backend.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)]
    )
