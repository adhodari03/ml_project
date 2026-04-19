"""
BaseAgent — Abstract base class for all specialist agents.

Each agent receives:
  - A user query (text)
  - Retrieved context from RAG (if available)
  - Optional metadata (images, files, etc.)

And returns:
  - A response dict with 'response' text, 'agent' name, and 'pipeline' telemetry
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    """Abstract base class for NexusAI specialist agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """
        Execute the agent's task.

        Args:
            query: The user's input text
            context: Retrieved RAG context (list of dicts with 'text' and 'similarity')
            metadata: Additional data (image_path, file_path, etc.)

        Returns:
            dict with 'response', 'agent', 'pipeline', etc.
        """
        pass

    def format_context(self, context: list) -> str:
        """Format retrieved context for injection into LLM prompt."""
        if not context:
            return ""

        formatted = "### Retrieved Context (from memory):\n"
        for i, item in enumerate(context, 1):
            sim = item.get("similarity", 0)
            text = item.get("text", "")
            formatted += f"\n[{i}] (similarity: {sim:.3f})\n{text}\n"
        formatted += "\n### End of Retrieved Context\n"
        return formatted
