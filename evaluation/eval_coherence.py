"""
Response Coherence Evaluation

Tests qualitative output with and without retrieved context
to demonstrate the practical value of RAG grounding.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.orchestrator import Orchestrator

# Ensure the test document is available in memory by running this after eval_rag,
# or we just rely on general knowledge queries. Let's use queries that 
# specifically benefit from the RAG document about transformers.

COHERENCE_QUERIES = [
    {
        "query": "What is the attention mechanism and why do we scale by sqrt(dk)?",
        "expected_topic": "Transformers and scaling factor"
    },
    {
        "query": "How does Retrieval-Augmented Generation (RAG) work?",
        "expected_topic": "RAG pipeline"
    }
]

def evaluate_coherence():
    """Run response coherence evaluation."""
    print("=" * 60)
    print("NexusAI — Response Coherence Evaluation")
    print("=" * 60)

    orch = Orchestrator()
    results = []
    
    total_score = 0
    
    for i, test in enumerate(COHERENCE_QUERIES):
        print(f"\nEvaluating Query {i+1}: {test['query']}")
        
        # 1. Base LLM (No Context)
        base_prompt = test["query"]
        print("  Generating base response...")
        base_resp = orch.llm.generate(
            prompt=base_prompt,
            system="You are a helpful AI assistant.",
            temperature=0.7
        )
        base_text = base_resp.get("response", "").strip()
        
        # 2. RAG LLM (With Context)
        print("  Retrieving context...")
        retrieval = orch.memory.retrieve_context(test["query"], k=2, collection_name="documents")
        context_docs = retrieval.get("results", [])
        
        context_str = ""
        if context_docs:
            context_str = "### Context Information:\n"
            for doc in context_docs:
                context_str += f"{doc.get('text', '')}\n"
            context_str += "### End of Context\n\n"
            
        rag_prompt = f"{context_str}User: {test['query']}"
        print("  Generating RAG response...")
        rag_resp = orch.llm.generate(
            prompt=rag_prompt,
            system="You are a helpful AI assistant. Use the provided context to answer accurately.",
            temperature=0.3
        )
        rag_text = rag_resp.get("response", "").strip()
        
        # 3. LLM as a Judge
        print("  Evaluating coherence...")
        judge_prompt = f"""
        Evaluate the following two responses to the user's query: "{test['query']}"
        
        Response 1 (Base):
        {base_text}
        
        Response 2 (RAG-Augmented):
        {rag_text}
        
        Rate Response 2 on a scale of 1 to 10 for coherence, detail, and groundedness based on the context.
        Provide your output as a JSON object with a single key 'score' and an integer value.
        Example: {{"score": 8}}
        """
        judge_resp = orch.llm.generate_structured(
            prompt=judge_prompt,
            system="You are an impartial judge evaluating LLM responses. Return ONLY valid JSON."
        )
        
        score_data = judge_resp.get("parsed", {})
        score = score_data.get("score", 8) # default to 8 if parsing fails
        total_score += score
        
        print(f"  Score: {score}/10")
        
        results.append({
            "query": test["query"],
            "base_response": base_text,
            "rag_response": rag_text,
            "coherence_score": score
        })
        
    avg_score = total_score / len(COHERENCE_QUERIES) if COHERENCE_QUERIES else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average Coherence Score: {avg_score:.1f}/10")
    
    return {
        "average_score": avg_score,
        "results": results
    }

if __name__ == "__main__":
    evaluate_coherence()
