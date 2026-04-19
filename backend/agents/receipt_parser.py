"""
ReceiptParserAgent — Receipt & Image Analysis Specialist

Uses LLaVA (vision-language model) to analyze receipt/invoice images
and extract structured purchase information.
"""

import time
from backend.agents.base import BaseAgent
from backend.models.vision import OllamaVision
from backend.models.llm import OllamaLLM
from backend.config import RECEIPT_AGENT_PROMPT


class ReceiptParserAgent(BaseAgent):
    """Analyzes receipt images and extracts structured data."""

    def __init__(self):
        super().__init__(
            name="Receipt Parser",
            description="Analyze receipts and invoices to extract purchase information"
        )
        self.vision = OllamaVision()
        self.llm = OllamaLLM()

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a receipt/image analysis request."""
        start_time = time.time()
        pipeline_log = []

        image_path = metadata.get("image_path") if metadata else None

        if image_path:
            # Use LLaVA for image analysis
            if "receipt" in query.lower() or "invoice" in query.lower():
                result = self.vision.analyze_receipt(image_path)
            else:
                result = self.vision.analyze_image(image_path, query)

            pipeline_log.extend(result.get("pipeline", []))

            return {
                "response": result.get("response", "Could not analyze image."),
                "agent": self.name,
                "pipeline": pipeline_log,
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        else:
            # No image provided — use LLM to explain
            context_str = self.format_context(context) if context else ""
            prompt = f"""{context_str}

The user seems to want receipt/image analysis but no image was provided.

User: {query}

Politely ask them to upload an image, or if they're asking about a previously analyzed receipt, use the context above to answer."""

            result = self.llm.generate(prompt, system=RECEIPT_AGENT_PROMPT)
            pipeline_log.extend(result.get("pipeline", []))

            return {
                "response": result.get("response", "Please upload a receipt image for analysis."),
                "agent": self.name,
                "pipeline": pipeline_log,
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }
