# NexusAI — Multimodal Agentic Assistant

A locally-hosted multimodal agentic assistant powered by LLM orchestration, RAG memory, and transformer-based models. Built as a Machine Learning course project demonstrating the practical application of transformer architectures, attention mechanisms, retrieval-augmented generation, and instruction-tuned language models.

## Features

- **🤖 LLM Orchestrator**: Llama 3 classifies user intent and routes to specialist agents
- **🎤 Voice Input**: Whisper ASR (encoder-decoder transformer) for speech-to-text
- **🖼️ Image Analysis**: LLaVA (CLIP + LLM) for vision-language tasks & receipt parsing
- **🧠 RAG Memory**: ChromaDB vector database with persistent conversational memory
- **📋 Task Management**: Natural language task CRUD operations
- **📄 Document Q&A**: Upload documents and ask questions (RAG-powered)
- **🐛 Code Debugging**: Multi-language code analysis and error fixing
- **🔬 ML Pipeline Visualizer**: Real-time visualization of the transformer pipeline

## Architecture

```
User Input → Whisper/LLaVA (if voice/image) → LLM Orchestrator (Llama 3)
           → Intent Classification → RAG Context Retrieval (ChromaDB)
           → Agent Routing → Specialist Agent Execution → Response
```

### Models (all local, via Ollama)
| Model | Role | Architecture |
|-------|------|-------------|
| Llama 3 (8B) | Orchestrator & Chat | Decoder-only Transformer (32 layers, GQA) |
| LLaVA | Image Analysis | CLIP ViT-L/14 + LLM Decoder |
| Whisper (small) | Speech-to-Text | Encoder-Decoder Transformer (12+12 layers) |
| nomic-embed-text | Embeddings | BERT-like Encoder (137M) |

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama server
ollama serve
```

### 2. Pull Models (new terminal)

```bash
ollama pull llama3
ollama pull llava
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies

```bash
cd ML_Project
pip install -r backend/requirements.txt
```

### 4. Run the Server

```bash
python -m backend.main
```

Open http://localhost:8000 in your browser.

## ML Pipeline Visualizer

The dashboard includes a real-time ML Pipeline Inspector showing:

- **Pipeline View**: Step-by-step breakdown of each inference call
- **Architecture View**: Interactive Llama 3 transformer architecture diagram
- **Attention View**: Simulated attention weight matrix with causal mask
- **Token Flow**: Visualize tokenization → embedding → transformer → output
- **RAG View**: Document chunking → embedding → similarity search pipeline

## Evaluation

```bash
# Routing accuracy evaluation
python -m evaluation.eval_routing

# RAG retrieval quality evaluation
python -m evaluation.eval_rag
```

## Project Structure

```
ML_Project/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Configuration
│   ├── orchestrator.py      # LLM-based intent classification & routing
│   ├── memory.py            # ChromaDB RAG memory layer
│   ├── models/
│   │   ├── llm.py           # Llama 3 interface (Ollama)
│   │   ├── vision.py        # LLaVA interface (Ollama)
│   │   └── speech.py        # Whisper ASR interface
│   └── agents/
│       ├── base.py           # Abstract base agent
│       ├── task_manager.py   # Task management agent
│       ├── receipt_parser.py # Receipt parsing agent (LLaVA)
│       ├── doc_qa.py         # Document Q&A agent (RAG)
│       └── code_debugger.py  # Code debugging agent
├── frontend/
│   ├── index.html           # Dashboard
│   ├── index.css            # Design system
│   └── app.js               # Application logic
├── evaluation/
│   ├── test_suite.json      # 20 test cases
│   ├── eval_routing.py      # Routing accuracy evaluation
│   └── eval_rag.py          # RAG retrieval evaluation
└── data/
    ├── chroma_db/           # Vector database (persistent)
    └── uploads/             # Uploaded files
```

## Key ML Concepts Demonstrated

1. **Transformer Architecture**: Decoder-only (Llama 3), Encoder-Decoder (Whisper)
2. **Attention Mechanism**: Scaled dot-product, multi-head, causal masking, GQA
3. **Instruction Tuning**: Why Llama 3 can follow structured output formats
4. **RAG**: Dense retrieval, cosine similarity, context grounding
5. **Vision-Language Models**: CLIP visual encoder + LLM decoder fusion
6. **ASR**: Audio → Mel spectrogram → Encoder → Decoder → Text

## License

MIT — Educational project for Machine Learning coursework.
