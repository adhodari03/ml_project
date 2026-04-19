"""
StudyBuddyAgent — Socratic Tutor and Academic Assistant
"""

import time
from backend.agents.base import BaseAgent
from backend.models.llm import OllamaLLM
from backend.config import STUDY_BUDDY_AGENT_PROMPT


class StudyBuddyAgent(BaseAgent):
    """Socratic Tutor and Study Assistant."""

    def __init__(self, memory=None):
        super().__init__(
            name="Study Buddy",
            description="A Socratic tutor that helps students learn concepts, summarize notes, and prepare for exams."
        )
        self.llm = OllamaLLM()
        self.memory = memory

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a study task."""
        start_time = time.time()
        pipeline_log = []

        # Grounding with any provided context (like uploaded class notes or conversation history)
        context_str = ""
        if context:
            context_str = self.format_context(context)

        prompt = f"{context_str}\n\nStudent Request: {query}"

        # Generate response using the strict Socratic prompt
        result = self.llm.generate(prompt, system=STUDY_BUDDY_AGENT_PROMPT)
        pipeline_log.extend(result.get("pipeline", []))

        # Add agent-specific telemetry
        pipeline_log.append({
            "stage": "socratic_tutoring",
            "description": "The Study Buddy agent applies pedagogical models (Feynman technique, Socratic questioning) to guide the student towards the answer rather than providing it directly.",
            "duration_ms": 0
        })

        return {
            "response": result.get("response", "Could not generate answer."),
            "agent": self.name,
            "pipeline": pipeline_log,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }
