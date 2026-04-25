"""
MemoryStore — ChromaDB RAG Memory Layer

Implements Retrieval-Augmented Generation (RAG) using ChromaDB as the vector store.

RAG Pipeline:
  1. Documents/conversations → Chunking → Embedding (dense vectors)
  2. Vectors stored in ChromaDB with metadata
  3. At query time: query → embed → cosine similarity search → top-k retrieval
  4. Retrieved chunks injected as context into LLM prompt

Cosine Similarity:
  sim(u, v) = (u · v) / (||u|| × ||v||)

This enables the assistant to "remember" past conversations and retrieve
relevant document passages without exceeding the LLM's context window.
"""

import time
import uuid
import chromadb
from datetime import datetime
from typing import List, Optional
from backend.config import (
    CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    COLLECTION_CONVERSATIONS, COLLECTION_DOCUMENTS, COLLECTION_TASKS,
    OLLAMA_BASE_URL, EMBEDDING_MODEL
)


class MemoryStore:
    """ChromaDB-backed vector memory for RAG."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._setup_collections()
        self._pipeline_log = []

    def _setup_collections(self):
        """Initialize ChromaDB collections."""
        self.conversations = self.client.get_or_create_collection(
            name=COLLECTION_CONVERSATIONS,
            metadata={"hnsw:space": "cosine"}
        )
        self.documents = self.client.get_or_create_collection(
            name=COLLECTION_DOCUMENTS,
            metadata={"hnsw:space": "cosine"}
        )
        self.tasks_collection = self.client.get_or_create_collection(
            name=COLLECTION_TASKS,
            metadata={"hnsw:space": "cosine"}
        )

    def clear_conversations(self):
        """Wipe the local conversation vector memory for a fresh start."""
        try:
            self.client.delete_collection(name=COLLECTION_CONVERSATIONS)
            self.conversations = self.client.create_collection(
                name=COLLECTION_CONVERSATIONS,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector using Ollama's nomic-embed-text model.

        nomic-embed-text produces 768-dimensional dense vectors trained
        with contrastive learning to place semantically similar texts
        close together in the vector space.
        """
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for embedding.

        Chunking strategy: Fixed-size windows with overlap to preserve
        context at chunk boundaries. Each chunk becomes one vector in
        the database.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - CHUNK_OVERLAP
        return chunks

    def store_memory(self, text: str, role: str = "user", metadata: dict = None) -> dict:
        """
        Embed and store a conversation message in the vector database.

        Pipeline:
          text → embedding model → 768-dim vector → ChromaDB (HNSW index)
        """
        self._pipeline_log = []
        start_time = time.time()

        # ── Stage 1: Text Embedding ──
        embed_start = time.time()
        embedding = self._get_embedding(text)

        self._pipeline_log.append({
            "stage": "text_embedding",
            "description": "Input text is tokenized and passed through nomic-embed-text (137M params). The model uses a BERT-like encoder with mean pooling over token embeddings to produce a single 768-dim vector.",
            "details": {
                "model": EMBEDDING_MODEL,
                "output_dim": len(embedding),
                "input_length": len(text),
                "pooling": "mean over token embeddings",
                "training": "Contrastive learning (InfoNCE loss)",
            },
            "duration_ms": round((time.time() - embed_start) * 1000, 2)
        })

        # ── Stage 2: Vector Indexing ──
        index_start = time.time()
        doc_id = str(uuid.uuid4())
        meta = {
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})

        self.conversations.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta]
        )

        self._pipeline_log.append({
            "stage": "vector_indexing",
            "description": "The embedding vector is inserted into ChromaDB's HNSW (Hierarchical Navigable Small World) index. HNSW is a graph-based approximate nearest neighbor (ANN) algorithm that enables O(log n) similarity search.",
            "details": {
                "index_type": "HNSW",
                "distance_metric": "cosine",
                "collection": COLLECTION_CONVERSATIONS,
                "total_vectors": self.conversations.count(),
            },
            "duration_ms": round((time.time() - index_start) * 1000, 2)
        })

        return {
            "id": doc_id,
            "pipeline": self._pipeline_log,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def retrieve_context(self, query: str, k: int = None, collection_name: str = None) -> dict:
        """
        Retrieve the k most similar documents/messages for a query.

        Pipeline:
          query → embed → cosine similarity search → top-k results
        """
        k = k or TOP_K_RETRIEVAL
        self._pipeline_log = []
        start_time = time.time()

        # Select collection
        if collection_name == "documents":
            collection = self.documents
        elif collection_name == "tasks":
            collection = self.tasks_collection
        else:
            collection = self.conversations

        if collection.count() == 0:
            return {"results": [], "pipeline": [], "total_time_ms": 0}

        if not query or not query.strip():
            return {"results": [], "pipeline": [], "total_time_ms": 0}

        # ── Stage 1: Query Embedding ──
        embed_start = time.time()
        query_embedding = self._get_embedding(query)

        self._pipeline_log.append({
            "stage": "query_embedding",
            "description": "The search query is embedded into the same 768-dim vector space as stored documents using nomic-embed-text, enabling meaningful distance comparisons.",
            "details": {
                "query_length": len(query),
                "embedding_dim": len(query_embedding),
            },
            "duration_ms": round((time.time() - embed_start) * 1000, 2)
        })

        # ── Stage 2: Similarity Search ──
        search_start = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count())
        )

        self._pipeline_log.append({
            "stage": "similarity_search",
            "description": "The HNSW index performs approximate nearest neighbor search using cosine similarity. It traverses the graph from entry point to the query's neighborhood, comparing sim(q, v) = (q·v)/(||q||·||v||).",
            "details": {
                "k": k,
                "candidates_searched": collection.count(),
                "results_returned": len(results["documents"][0]) if results["documents"] else 0,
                "distances": results["distances"][0] if results["distances"] else [],
            },
            "duration_ms": round((time.time() - search_start) * 1000, 2)
        })

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                formatted.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(1 - dist, 4),  # Convert distance to similarity
                    "rank": i + 1
                })

        return {
            "results": formatted,
            "pipeline": self._pipeline_log,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def store_document(self, file_path: str, content: str = None) -> dict:
        """
        Ingest a document: chunk it, embed each chunk, and store in ChromaDB.

        For PDF files, text is extracted first. The document is then split
        into overlapping chunks to maintain context at boundaries.
        """
        from pathlib import Path
        self._pipeline_log = []
        start_time = time.time()

        path = Path(file_path)

        # Extract text from file
        if content is None:
            if path.suffix.lower() == ".pdf":
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif path.suffix.lower() in [".txt", ".md", ".py", ".js", ".html", ".css"]:
                content = path.read_text(encoding="utf-8", errors="ignore")
            else:
                return {"error": f"Unsupported file type: {path.suffix}", "pipeline": []}

        # ── Stage 1: Chunking ──
        chunk_start = time.time()
        chunks = self._chunk_text(content)

        self._pipeline_log.append({
            "stage": "document_chunking",
            "description": f"Document is split into overlapping chunks of {CHUNK_SIZE} characters with {CHUNK_OVERLAP} character overlap. Each chunk becomes an independent retrieval unit.",
            "details": {
                "document": path.name,
                "total_chars": len(content),
                "num_chunks": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
            },
            "duration_ms": round((time.time() - chunk_start) * 1000, 2)
        })

        # ── Stage 2: Batch Embedding & Indexing ──
        embed_start = time.time()
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = [self._get_embedding(chunk) for chunk in chunks]

        metadatas = [{
            "source": path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat(),
        } for i in range(len(chunks))]

        self.documents.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        self._pipeline_log.append({
            "stage": "batch_embedding_indexing",
            "description": f"All {len(chunks)} chunks are individually embedded via nomic-embed-text and inserted into the HNSW index. Each chunk is now independently searchable by semantic similarity.",
            "details": {
                "chunks_embedded": len(chunks),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "total_vectors_in_collection": self.documents.count(),
            },
            "duration_ms": round((time.time() - embed_start) * 1000, 2)
        })

        return {
            "chunks_stored": len(chunks),
            "document": path.name,
            "pipeline": self._pipeline_log,
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "error": None
        }

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        return {
            "conversations": self.conversations.count(),
            "documents": self.documents.count(),
            "tasks": self.tasks_collection.count(),
        }

    def clear_collection(self, name: str):
        """Clear a specific collection."""
        self.client.delete_collection(name)
        self._setup_collections()
