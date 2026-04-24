"""
FinanceAgent — Personal Finance & Expense Specialist

Handles multi-step financial tasks: receipt parsing (vision), math calculations,
logging expenses via TaskManagerAgent, and answering specific spending inquiries
using RAG context.
"""

import time
from backend.agents.base import BaseAgent
from backend.models.vision import OllamaVision
from backend.models.llm import OllamaLLM
from backend.agents.task_manager import TaskManagerAgent
from backend.config import FINANCE_AGENT_PROMPT

class FinanceAgent(BaseAgent):
    """Handles multi-step finance and expense tracking."""

    def __init__(self, memory):
        super().__init__(
            name="Personal Finance",
            description="Analyze receipts, calculate tips, log expenses, and compare spending"
        )
        self.vision = OllamaVision()
        self.llm = OllamaLLM()
        self.task_manager = TaskManagerAgent()
        self.memory = memory

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a multi-step finance request."""
        start_time = time.time()
        pipeline_log = []

        image_path = metadata.get("image_path") if metadata else None
        receipt_text = ""

        # Step 1: LLaVA Vision extraction
        if image_path:
            vision_start = time.time()
            vision_result = self.vision.analyze_receipt(image_path)
            receipt_text = vision_result.get("response", "Could not analyze receipt.")
            
            pipeline_log.append({
                "stage": "image_preprocessing",
                "description": "FinanceAgent called LLaVA vision model to extract raw receipt information first.",
                "details": {"receipt_extracted": receipt_text[:100] + "..."},
                "duration_ms": round((time.time() - vision_start) * 1000, 2)
            })
            pipeline_log.extend(vision_result.get("pipeline", []))

        # Step 2: Assemble prompt for LLM
        context_str = self.format_context(context) if context else "No past memory found."
        
        prompt = f"""
Receipt Analysis (if applicable):
{receipt_text}

Context from Past Conversations & Memories:
{context_str}

User Request: {query}

Instructions:
1. Review the Receipt Analysis. If costs and tips need calculation, calculate them accurately (e.g., add 18% tip).
2. Generate exactly one JSON action block to log the expense. Use the 'create' action, and log it to the 'Task Manager' as an expense.
Format exactly like this:
```json
{{"action": "create", "task": "Dining expense: $TOTAL_WITH_TIP", "priority": "high"}}
```
3. Following the JSON block, answer the conversational part of the user's request (e.g., comparing spending to past memory).
"""

        llm_start = time.time()
        result = self.llm.generate(prompt, system=FINANCE_AGENT_PROMPT, temperature=0.2)
        response_text = result.get("response", "")
        pipeline_log.extend(result.get("pipeline", []))

        # Step 3: Extract and execute Task Action
        action_result = self.task_manager._parse_and_execute(response_text, query)
        if action_result:
            pipeline_log.append({
                "stage": "task_execution",
                "description": f"FinanceAgent delegated task creation to TaskManagerAgent: {action_result.get('action', 'unknown')}",
                "details": action_result,
                "duration_ms": round((time.time() - llm_start) * 1000, 2)
            })

            try:
                if "```json" in response_text:
                    parts = response_text.split("```json")
                    pre_json = parts[0]
                    post_json = parts[1].split("```", 1)[1] if "```" in parts[1] else parts[1]
                    response_text = f"{pre_json.strip()}\n\n*Task successfully logged!* ✅\n\n{post_json.strip()}"
            except Exception:
                pass

        return {
            "response": response_text.strip() or "Processed finance request.",
            "agent": self.name,
            "pipeline": pipeline_log,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }
