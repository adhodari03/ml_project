"""
DocQAAgent — Document Q&A Specialist (RAG-Powered)

Answers questions about uploaded documents by:
  1. Retrieving relevant document chunks from ChromaDB
  2. Injecting retrieved context into the LLM prompt
  3. Generating grounded responses with source citations

This is the core RAG application — demonstrating how retrieval
augments the LLM's parametric knowledge with external evidence.
"""

import time
from backend.agents.base import BaseAgent
from backend.models.llm import OllamaLLM
from backend.memory import MemoryStore
from backend.config import DOC_QA_AGENT_PROMPT


class DocQAAgent(BaseAgent):
    """RAG-powered document question answering."""

    def __init__(self, memory: MemoryStore):
        super().__init__(
            name="Document Q&A",
            description="Answer questions about uploaded documents using RAG"
        )
        self.llm = OllamaLLM()
        self.memory = memory

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a document Q&A request."""
        start_time = time.time()
        pipeline_log = []

        # ── Stage 1: Retrieve relevant document chunks ──
        retrieval_result = self.memory.retrieve_context(
            query, k=5, collection_name="documents"
        )
        pipeline_log.extend(retrieval_result.get("pipeline", []))
        doc_context = retrieval_result.get("results", [])

        # Also check conversation history for context
        if context:
            doc_context.extend(context)

        # ── Stage 2: Build grounded prompt ──
        context_str = self.format_context(doc_context)

        if not doc_context:
            prompt = f"""No documents have been uploaded yet, or no relevant content was found.

User question: {query}

Let the user know they need to upload a document first, or that the question might not relate to any uploaded document."""
        else:
            prompt = f"""{context_str}

Based ONLY on the retrieved context above, answer the following question.
If the context doesn't contain enough information, clearly state that.
Cite specific parts of the context in your answer.

Question: {query}"""

        # ── Stage 3: Generate grounded response ──
        result = self.llm.generate(prompt, system=DOC_QA_AGENT_PROMPT)
        pipeline_log.extend(result.get("pipeline", []))

        # Add RAG summary to pipeline
        pipeline_log.append({
            "stage": "rag_grounding",
            "description": "Response is grounded in retrieved document chunks rather than the LLM's parametric knowledge. This reduces hallucination and enables the model to answer questions about documents it was never trained on.",
            "details": {
                "chunks_retrieved": len(doc_context),
                "sources": list(set(
                    item.get("metadata", {}).get("source", "conversation")
                    for item in doc_context
                )),
                "avg_similarity": round(
                    sum(item.get("similarity", 0) for item in doc_context) / max(len(doc_context), 1), 4
                ),
            },
            "duration_ms": 0
        })

        return {
            "response": result.get("response", "Could not generate answer."),
            "agent": self.name,
            "pipeline": pipeline_log,
            "sources": doc_context,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def ingest_document(self, file_path: str, content: str = None) -> dict:
        """Ingest a document into the RAG store."""
        return self.memory.store_document(file_path, content)
