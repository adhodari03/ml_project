"""
RAG Retrieval Quality Evaluation

Tests the retrieval quality of the RAG memory layer by:
1. Ingesting a test document
2. Running a set of queries
3. Measuring precision and relevance of retrieved chunks

Metrics:
  - Retrieval Precision: % of retrieved chunks that are relevant
  - Mean Reciprocal Rank (MRR): How high the first relevant result appears
  - Mean Similarity Score: Average cosine similarity of top-k results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.memory import MemoryStore


# Test document content (simulating a document about transformers)
TEST_DOCUMENT = """
Chapter 1: Introduction to Transformers
The transformer architecture was introduced by Vaswani et al. in 2017 in the paper
"Attention Is All You Need". It replaced recurrent neural networks (RNNs) as the
dominant architecture for sequence modeling tasks in NLP. The key innovation is the
self-attention mechanism, which allows the model to process all positions in a
sequence in parallel, rather than sequentially like RNNs.

Chapter 2: The Attention Mechanism
The scaled dot-product attention operates on queries (Q), keys (K), and values (V).
The attention weights are computed as softmax(QK^T / sqrt(dk)), where dk is the
dimension of the key vectors. This scaling factor prevents the dot products from
growing too large in high dimensions, which would push the softmax into regions
with very small gradients. Multi-head attention runs multiple attention functions
in parallel, each learning different patterns.

Chapter 3: Encoder-Decoder Architecture
The original transformer uses an encoder-decoder structure. The encoder processes
the input sequence and produces contextualized representations. The decoder generates
the output sequence one token at a time, using both self-attention over previously
generated tokens and cross-attention to the encoder's output. Modern LLMs like GPT
and Llama use decoder-only architectures.

Chapter 4: Training and Fine-tuning
Pre-training involves predicting the next token in a large text corpus (causal
language modeling). Fine-tuning adapts the model to specific tasks. Instruction
tuning trains the model on (instruction, response) pairs. RLHF (Reinforcement
Learning from Human Feedback) further aligns the model's outputs with human
preferences using a reward model.

Chapter 5: Retrieval-Augmented Generation
RAG combines retrieval with generation. Documents are chunked, embedded into vectors,
and stored in a vector database. At inference time, relevant chunks are retrieved
using similarity search and injected into the LLM's prompt as context. This reduces
hallucination and enables knowledge updates without retraining.
"""

TEST_QUERIES = [
    {
        "query": "What is the attention mechanism?",
        "relevant_chapter": "Chapter 2",
        "description": "Direct question about attention"
    },
    {
        "query": "Why do we scale by sqrt(dk)?",
        "relevant_chapter": "Chapter 2",
        "description": "Specific detail about scaling"
    },
    {
        "query": "How does RAG work?",
        "relevant_chapter": "Chapter 5",
        "description": "RAG pipeline question"
    },
    {
        "query": "What is instruction tuning?",
        "relevant_chapter": "Chapter 4",
        "description": "Training methodology question"
    },
    {
        "query": "What replaced RNNs?",
        "relevant_chapter": "Chapter 1",
        "description": "Historical context question"
    },
    {
        "query": "Explain cross-attention in transformers",
        "relevant_chapter": "Chapter 3",
        "description": "Architecture detail question"
    },
]


def evaluate_rag():
    """Run RAG retrieval quality evaluation."""
    print("=" * 60)
    print("NexusAI — RAG Retrieval Quality Evaluation")
    print("=" * 60)

    # Initialize memory store
    memory = MemoryStore()

    # Clear existing documents for clean evaluation
    try:
        memory.clear_collection("documents")
    except:
        pass

    # Ingest test document
    print("\n1. Ingesting test document...")
    tmp_path = Path("/tmp/nexusai_test_doc.txt")
    tmp_path.write_text(TEST_DOCUMENT)
    result = memory.store_document(str(tmp_path))
    print(f"   Chunks stored: {result.get('chunks_stored', 0)}")
    print(f"   Time: {result.get('total_time_ms', 0):.0f}ms")

    # Run queries
    print("\n2. Running queries...")
    print("-" * 60)

    total_mrr = 0
    total_sim = 0
    total_precision = 0
    valid_queries = 0

    for test in TEST_QUERIES:
        print(f"\n   Query: {test['query']}")
        print(f"   Expected: {test['relevant_chapter']}")

        retrieval = memory.retrieve_context(test["query"], k=3, collection_name="documents")
        results = retrieval.get("results", [])

        if not results:
            print("   ⚠️  No results returned")
            continue

        valid_queries += 1

        # Check if relevant chapter is in results
        relevant_found = False
        first_relevant_rank = None
        relevant_count = 0

        for i, r in enumerate(results):
            text = r.get("text", "")
            sim = r.get("similarity", 0)
            is_relevant = test["relevant_chapter"].lower() in text.lower()

            if is_relevant:
                relevant_count += 1
                if not relevant_found:
                    first_relevant_rank = i + 1
                    relevant_found = True

            status = "✅" if is_relevant else "  "
            print(f"   {status} [{i+1}] sim={sim:.4f} | {text[:60]}...")

        # Metrics
        mrr = 1 / first_relevant_rank if first_relevant_rank else 0
        precision = relevant_count / len(results) if results else 0
        avg_sim = sum(r.get("similarity", 0) for r in results) / len(results)

        total_mrr += mrr
        total_precision += precision
        total_sim += avg_sim

        print(f"   MRR: {mrr:.3f} | Precision: {precision:.3f} | Avg Sim: {avg_sim:.4f}")

    # Summary
    n = max(valid_queries, 1)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Queries evaluated: {valid_queries}/{len(TEST_QUERIES)}")
    print(f"  Mean Reciprocal Rank (MRR): {total_mrr / n:.3f}")
    print(f"  Mean Precision@3: {total_precision / n:.3f}")
    print(f"  Mean Similarity Score: {total_sim / n:.4f}")

    # Cleanup
    tmp_path.unlink(missing_ok=True)

    return {
        "mrr": total_mrr / n,
        "precision": total_precision / n,
        "avg_similarity": total_sim / n,
        "queries_evaluated": valid_queries
    }


if __name__ == "__main__":
    evaluate_rag()
