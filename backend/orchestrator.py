"""
Orchestrator — LLM-Based Intent Classification & Agent Routing

The orchestrator is the "brain" of NexusAI. It uses Llama 3 to:
  1. Classify the user's intent into one of 5 categories
  2. Route the request to the appropriate specialist agent
  3. Retrieve relevant context from RAG memory
  4. Return the agent's response with full pipeline telemetry

The intent classification leverages instruction tuning — Llama 3 has been
fine-tuned on (instruction, response) pairs, enabling it to follow the
structured JSON output format reliably. This would NOT work with a raw
pre-trained model.
"""

import time
from typing import Optional
from backend.models.llm import OllamaLLM
from backend.models.vision import OllamaVision
from backend.memory import MemoryStore
from backend.agents.task_manager import TaskManagerAgent
from backend.agents.receipt_parser import ReceiptParserAgent
from backend.agents.doc_qa import DocQAAgent
from backend.agents.code_debugger import CodeDebuggerAgent
from backend.agents.study_buddy import StudyBuddyAgent
from backend.config import (
    ORCHESTRATOR_SYSTEM_PROMPT, GENERAL_CHAT_PROMPT
)


class Orchestrator:
    """
    LLM-based orchestrator that classifies intent and routes to specialist agents.

    Flow:
      User Input → Intent Classification (Llama 3) → Agent Selection
                → RAG Context Retrieval → Agent Execution → Response
    """

    def __init__(self):
        self.llm = OllamaLLM()
        self.memory = MemoryStore()

        # Initialize specialist agents
        self.agents = {
            "task_management": TaskManagerAgent(),
            "receipt_parsing": ReceiptParserAgent(),
            "document_qa": DocQAAgent(self.memory),
            "code_debugging": CodeDebuggerAgent(),
            "study_buddy": StudyBuddyAgent(self.memory),
        }

    def process_message(self, message: str, metadata: dict = None, agent_override: str = None) -> dict:
        """
        Process user message through the full pipeline:
        1. Classify Intent (or use agent_override)
        2. Retrieve Context (RAG)
        3. Route to specific Agent
        4. Store in Memory
        """
        start_time = time.time()
        full_pipeline = []

        # ── Stage 1: Intent Classification ──
        classify_start = time.time()
        
        if agent_override and (agent_override in self.agents or agent_override == "general_chat"):
            intent = agent_override
            confidence = 1.0
            reasoning = f"User explicitly selected {agent_override} via sidebar."
            
            full_pipeline.append({
                "stage": "intent_classification",
                "description": "Classification bypassed. User explicitly routed message to a specific agent.",
                "details": {
                    "classified_intent": intent,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "available_agents": list(self.agents.keys()) + ["general_chat"],
                    "entities": {},
                },
                "duration_ms": 0
            })
        else:
            classification = self.llm.generate_structured(
                prompt=message,
                system=ORCHESTRATOR_SYSTEM_PROMPT
            )

            intent_data = classification.get("parsed", {})
            intent = intent_data.get("intent", "general_chat")
            confidence = intent_data.get("confidence", 0.5)
            reasoning = intent_data.get("reasoning", "")

            full_pipeline.append({
                "stage": "intent_classification",
                "description": "The orchestrator uses Llama 3 (instruction-tuned) to classify the user's intent into one of 5 categories. The model outputs structured JSON thanks to instruction tuning — it has learned to follow specific output formats during fine-tuning.",
                "details": {
                    "classified_intent": intent,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "available_agents": list(self.agents.keys()) + ["general_chat"],
                    "entities": intent_data.get("extracted_entities", {}),
                },
                "duration_ms": round((time.time() - classify_start) * 1000, 2)
            })

            # Add LLM pipeline from classification
            full_pipeline.extend(classification.get("pipeline", []))

        # ── Stage 2: RAG Context Retrieval ──
        retrieval_start = time.time()
        retrieval = self.memory.retrieve_context(message)
        context = retrieval.get("results", [])

        full_pipeline.append({
            "stage": "rag_retrieval",
            "description": "Before agent execution, relevant context is retrieved from the vector memory. The user's message is embedded and compared against stored conversation history using cosine similarity in ChromaDB's HNSW index.",
            "details": {
                "query": message[:100] + "..." if len(message) > 100 else message,
                "results_found": len(context),
                "top_similarity": context[0]["similarity"] if context else None,
            },
            "duration_ms": round((time.time() - retrieval_start) * 1000, 2)
        })

        full_pipeline.extend(retrieval.get("pipeline", []))

        # ── Stage 3: Agent Routing & Execution ──
        route_start = time.time()
        agent_result = self._route_to_agent(intent, message, context, metadata)

        full_pipeline.append({
            "stage": "agent_routing",
            "description": f"Based on the classified intent '{intent}' (confidence: {confidence:.2f}), the request is routed to the '{agent_result.get('agent', 'General Chat')}' agent. The agent receives the query plus {len(context)} retrieved context items.",
            "details": {
                "selected_agent": agent_result.get("agent", "General Chat"),
                "intent": intent,
                "context_items_provided": len(context),
            },
            "duration_ms": round((time.time() - route_start) * 1000, 2)
        })

        full_pipeline.extend(agent_result.get("pipeline", []))

        # ── Stage 4: Memory Storage ──
        memory_start = time.time()

        # Store user message
        self.memory.store_memory(message, role="user", metadata={
            "intent": intent, "confidence": confidence
        })

        # Store assistant response
        response_text = agent_result.get("response", "")
        self.memory.store_memory(response_text, role="assistant", metadata={
            "agent": agent_result.get("agent", "General Chat"),
            "intent": intent
        })

        full_pipeline.append({
            "stage": "memory_storage",
            "description": "Both the user message and assistant response are embedded and stored in ChromaDB. This builds the persistent conversational memory, enabling future queries to retrieve relevant past interactions.",
            "details": {
                "stored_messages": 2,
                "total_memories": self.memory.get_stats()["conversations"],
            },
            "duration_ms": round((time.time() - memory_start) * 1000, 2)
        })

        total_time = time.time() - start_time

        return {
            "response": response_text,
            "agent": agent_result.get("agent", "General Chat"),
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "pipeline": full_pipeline,
            "total_time_ms": round(total_time * 1000, 2),
            "memory_stats": self.memory.get_stats()
        }

    def _route_to_agent(self, intent: str, message: str, context: list, metadata: dict = None) -> dict:
        """Route to the appropriate specialist agent based on classified intent."""

        if intent in self.agents:
            agent = self.agents[intent]
            return agent.execute(message, context=context, metadata=metadata)

        # Default: General Chat
        return self._general_chat(message, context, metadata)

    def _general_chat(self, message: str, context: list, metadata: dict = None) -> dict:
        """Handle general conversation with RAG context."""
        start_time = time.time()

        context_str = ""
        if context:
            context_str = "### Conversation History:\n"
            for item in context:
                role = item.get("metadata", {}).get("role", "unknown")
                text = item.get("text", "")
                context_str += f"[{role}]: {text}\n"
            context_str += "### End of History\n\n"

        prompt = f"""{context_str}User: {message}"""

        image_path = metadata.get("image_path") if metadata else None
        if image_path:
            vision_model = OllamaVision()
            result = vision_model.analyze_image(image_path, prompt)
        else:
            result = self.llm.generate(prompt, system=GENERAL_CHAT_PROMPT)

        return {
            "response": result.get("response", "I'm not sure how to help with that."),
            "agent": "General Chat",
            "pipeline": result.get("pipeline", []),
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def get_health(self) -> dict:
        """Check health of all components."""
        return {
            "llm": self.llm.check_health(),
            "memory": self.memory.get_stats(),
            "agents": {name: {"status": "ready"} for name in self.agents},
        }
