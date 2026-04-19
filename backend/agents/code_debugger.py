"""
CodeDebuggerAgent — Code Debugging Specialist

Analyzes code snippets, identifies bugs, explains errors,
and suggests fixes across multiple programming languages.
"""

import time
from backend.agents.base import BaseAgent
from backend.models.llm import OllamaLLM
from backend.config import CODE_DEBUG_AGENT_PROMPT


class CodeDebuggerAgent(BaseAgent):
    """Analyzes code, identifies bugs, and suggests fixes."""

    def __init__(self):
        super().__init__(
            name="Code Debugger",
            description="Debug code, explain errors, and suggest fixes"
        )
        self.llm = OllamaLLM()

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a code debugging request."""
        start_time = time.time()
        pipeline_log = []

        context_str = self.format_context(context) if context else ""

        # Detect programming language hints
        lang_hints = self._detect_language(query)

        prompt = f"""{context_str}

{f"Detected language: {lang_hints}" if lang_hints else ""}

User request: {query}

Please:
1. Identify any bugs or issues in the code
2. Explain the root cause of each issue
3. Provide the corrected code with explanations
4. Suggest any improvements or best practices"""

        result = self.llm.generate(prompt, system=CODE_DEBUG_AGENT_PROMPT)
        pipeline_log.extend(result.get("pipeline", []))

        return {
            "response": result.get("response", "Could not analyze code."),
            "agent": self.name,
            "pipeline": pipeline_log,
            "detected_language": lang_hints,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection from code patterns."""
        indicators = {
            "python": ["def ", "import ", "print(", "class ", "self.", "elif", "__init__", "lambda"],
            "javascript": ["const ", "let ", "var ", "function ", "=>", "console.log", "async ", "await "],
            "java": ["public static", "System.out", "void main", "class ", "import java"],
            "c++": ["#include", "std::", "cout", "cin", "int main"],
            "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE"],
            "html": ["<div", "<html", "<head", "<body", "<!DOCTYPE"],
            "css": ["margin:", "padding:", "display:", "color:", "font-size:"],
        }

        text_lower = text.lower()
        scores = {}
        for lang, keywords in indicators.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[lang] = score

        if scores:
            return max(scores, key=scores.get)
        return ""
