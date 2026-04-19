"""
TaskManagerAgent — Task Management Specialist

Handles creating, listing, updating, completing, and deleting tasks.
Tasks are persisted in a JSON file and also stored in ChromaDB for
semantic search (e.g., "what tasks do I have about machine learning?").
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from backend.agents.base import BaseAgent
from backend.models.llm import OllamaLLM
from backend.config import TASKS_FILE, TASK_AGENT_PROMPT


class TaskManagerAgent(BaseAgent):
    """Manages user tasks with natural language processing."""

    def __init__(self):
        super().__init__(
            name="Task Manager",
            description="Create, list, update, complete, and delete tasks"
        )
        self.llm = OllamaLLM()
        self.tasks_file = TASKS_FILE
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from persistent JSON storage."""
        if self.tasks_file.exists():
            with open(self.tasks_file, "r") as f:
                self.tasks = json.load(f)
        else:
            self.tasks = []

    def _save_tasks(self):
        """Save tasks to persistent JSON storage."""
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tasks_file, "w") as f:
            json.dump(self.tasks, f, indent=2, default=str)

    def execute(self, query: str, context: list = None, metadata: dict = None) -> dict:
        """Process a task management request."""
        start_time = time.time()
        pipeline_log = []

        # Build prompt with current tasks and context
        tasks_summary = self._get_tasks_summary()
        context_str = self.format_context(context) if context else ""

        prompt = f"""Current Tasks:
{tasks_summary}

{context_str}

User Request: {query}

Analyze the request and respond with a JSON action followed by a natural language response.
JSON action format:
```json
{{"action": "create|list|complete|update|delete", "task": "task description", "task_id": null_or_id, "priority": "low|medium|high"}}
```

Then provide a friendly response confirming the action."""

        # Get LLM response
        llm_start = time.time()
        result = self.llm.generate(prompt, system=TASK_AGENT_PROMPT, temperature=0.3)
        pipeline_log.extend(result.get("pipeline", []))

        response_text = result.get("response", "")

        # Try to parse and execute action
        action_result = self._parse_and_execute(response_text, query)
        if action_result:
            pipeline_log.append({
                "stage": "task_execution",
                "description": f"Executed task action: {action_result.get('action', 'unknown')}",
                "details": action_result,
                "duration_ms": round((time.time() - llm_start) * 1000, 2)
            })

        return {
            "response": response_text,
            "agent": self.name,
            "pipeline": pipeline_log,
            "tasks": self.tasks,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    def _parse_and_execute(self, response: str, original_query: str) -> Optional[dict]:
        """Parse LLM response for task actions and execute them."""
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                # Try to find JSON object
                import re
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    json_str = json_match.group()
                else:
                    return None

            action_data = json.loads(json_str.strip())
            action = action_data.get("action", "").lower()

            if action == "create":
                task = {
                    "id": len(self.tasks) + 1,
                    "text": action_data.get("task", original_query),
                    "priority": action_data.get("priority", "medium"),
                    "status": "pending",
                    "created": datetime.now().isoformat(),
                    "completed": None
                }
                self.tasks.append(task)
                self._save_tasks()
                return {"action": "create", "task": task}

            elif action == "complete":
                task_id = action_data.get("task_id")
                for task in self.tasks:
                    if task["id"] == task_id or (task_id is None and task["status"] == "pending"):
                        task["status"] = "completed"
                        task["completed"] = datetime.now().isoformat()
                        self._save_tasks()
                        return {"action": "complete", "task": task}

            elif action == "delete":
                task_id = action_data.get("task_id")
                self.tasks = [t for t in self.tasks if t["id"] != task_id]
                self._save_tasks()
                return {"action": "delete", "task_id": task_id}

            elif action == "list":
                return {"action": "list", "count": len(self.tasks)}

            return action_data

        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def _get_tasks_summary(self) -> str:
        """Get a formatted summary of current tasks."""
        if not self.tasks:
            return "No tasks currently."

        lines = []
        for task in self.tasks:
            status = "✅" if task["status"] == "completed" else "⬜"
            priority = task.get("priority", "medium")
            lines.append(f"  {status} [{task['id']}] ({priority}) {task['text']}")
        return "\n".join(lines)

    def get_all_tasks(self) -> list:
        """Return all tasks."""
        self._load_tasks()
        return self.tasks
