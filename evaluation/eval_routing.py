"""
Routing Accuracy Evaluation

Tests the orchestrator's intent classification accuracy against
the test suite. Measures precision, recall, and generates a
confusion matrix for each intent category.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.orchestrator import Orchestrator


def evaluate_routing():
    """Run routing accuracy evaluation."""
    # Load test suite
    test_file = Path(__file__).parent / "test_suite.json"
    with open(test_file) as f:
        test_cases = json.load(f)

    print("=" * 60)
    print("NexusAI — Routing Accuracy Evaluation")
    print("=" * 60)
    print(f"\nTest cases: {len(test_cases)}")
    print("-" * 60)

    # Initialize orchestrator
    orch = Orchestrator()

    results = []
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0

    for test in test_cases:
        print(f"\n[Test {test['id']}] {test['description']}")
        print(f"  Input: {test['input'][:80]}{'...' if len(test['input']) > 80 else ''}")
        print(f"  Expected: {test['expected_intent']}")

        try:
            # Classify intent only (don't run full pipeline)
            from backend.config import ORCHESTRATOR_SYSTEM_PROMPT
            classification = orch.llm.generate_structured(
                prompt=test["input"],
                system=ORCHESTRATOR_SYSTEM_PROMPT
            )
            intent_data = classification.get("parsed", {})
            predicted = intent_data.get("intent", "general_chat")
            confidence = intent_data.get("confidence", 0)

            is_correct = predicted == test["expected_intent"]
            if is_correct:
                correct += 1

            confusion[test["expected_intent"]][predicted] += 1

            results.append({
                "id": test["id"],
                "correct": is_correct,
                "expected": test["expected_intent"],
                "predicted": predicted,
                "confidence": confidence
            })

            status = "✅" if is_correct else "❌"
            print(f"  Predicted: {predicted} (confidence: {confidence:.2f}) {status}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                "id": test["id"],
                "correct": False,
                "expected": test["expected_intent"],
                "predicted": "error",
                "confidence": 0
            })

    # Summary
    accuracy = correct / len(test_cases) if test_cases else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")

    # Per-class metrics
    intents = sorted(set(t["expected_intent"] for t in test_cases))
    print(f"\n{'Intent':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)

    for intent in intents:
        tp = confusion[intent].get(intent, 0)
        fp = sum(confusion[other].get(intent, 0) for other in intents if other != intent)
        fn = sum(v for k, v in confusion[intent].items() if k != intent)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{intent:<20} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%}")

    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>20}", end="")
    for intent in intents:
        print(f"{intent[:8]:>10}", end="")
    print()

    for true_intent in intents:
        print(f"{true_intent:<20}", end="")
        for pred_intent in intents:
            count = confusion[true_intent].get(pred_intent, 0)
            print(f"{count:>10}", end="")
        print()

    # Save results
    output = {
        "accuracy": accuracy,
        "total": len(test_cases),
        "correct": correct,
        "results": results
    }

    output_file = Path(__file__).parent / "routing_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    evaluate_routing()
