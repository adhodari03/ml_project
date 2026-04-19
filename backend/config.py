"""
NexusAI Configuration
Central configuration for all model paths, API endpoints, and system settings.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
TASKS_FILE = DATA_DIR / "tasks.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure directories exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Ollama Configuration ───────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ─── Whisper Configuration ──────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

# ─── RAG Configuration ─────────────────────────────────────────
CHUNK_SIZE = 500          # Characters per chunk for document splitting
CHUNK_OVERLAP = 50        # Overlap between chunks
TOP_K_RETRIEVAL = 5       # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for retrieval

# ─── ChromaDB Collections ──────────────────────────────────────
COLLECTION_CONVERSATIONS = "conversations"
COLLECTION_DOCUMENTS = "documents"
COLLECTION_TASKS = "tasks"

# ─── LLM System Prompts ────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You are NexusAI, an intelligent orchestrator that classifies user intent and routes requests to the appropriate specialist agent.

Given a user message, classify it into EXACTLY ONE of these categories:
- "study_buddy": Answering academic questions, tutoring, summarizing notes, generating flashcards, explaining concepts
- "task_management": Creating, listing, updating, completing, or deleting tasks/todos/reminders
- "receipt_parsing": Analyzing receipts, invoices, or extracting purchase information from images
- "document_qa": Questions about uploaded documents, summarization, or document analysis
- "code_debugging": Code review, debugging, error explanation, or programming help
- "general_chat": General conversation, greetings, or questions not fitting other categories

You MUST respond with valid JSON in this exact format:
{
    "intent": "<category>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>",
    "extracted_entities": {<relevant extracted information>}
}"""

TASK_AGENT_PROMPT = """You are a Task Management specialist agent. Help users manage their tasks and to-do lists.
You can create, list, update, complete, and delete tasks. Be organized and helpful.
Always confirm actions taken and provide a clear summary of the current task state."""

RECEIPT_AGENT_PROMPT = """You are a Receipt Analysis specialist agent. Analyze receipt and invoice images to extract structured information.
Extract: vendor/store name, date, individual items with prices, subtotal, tax, total, payment method.
Present the information in a clear, organized format."""

DOC_QA_AGENT_PROMPT = """You are a Document Q&A specialist agent. Answer questions about uploaded documents using retrieved context.
Base your answers ONLY on the provided context. If the context doesn't contain enough information, say so clearly.
Always cite which part of the document your answer comes from."""

CODE_DEBUG_AGENT_PROMPT = """You are a Code Debugging specialist agent. Help users debug code, explain errors, and suggest fixes.
Analyze code carefully, identify bugs, explain the root cause, and provide corrected code.
Support multiple programming languages. Be thorough but concise."""

STUDY_BUDDY_AGENT_PROMPT = """You are a Study Buddy and Socratic Tutor. Your goal is to help students learn effectively.
Never give the direct answer to a homework or math problem immediately.
Instead, use the Socratic method: ask guiding questions, explain underlying concepts simply (like the Feynman technique), and encourage the student to arrive at the answer themselves.
If the student asks to summarize notes or create flashcards, do so in a highly structured, organized markdown format.
Always be encouraging, patient, and academic."""

GENERAL_CHAT_PROMPT = """You are NexusAI, a friendly and knowledgeable AI assistant.
Engage in helpful conversation while being informative and concise.
You have access to conversation history for context."""

# ─── Server Configuration ──────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
CORS_ORIGINS = ["*"]
