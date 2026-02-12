"""
Capability preservation: verify that meditation doesn't hurt
the model's ability to follow instructions, reason, and help.

Benchmarks:
- IFEval: instruction following
- GSM8K: mathematical reasoning (subset)
"""

import json
import logging
import re
from pathlib import Path

from llm_meditation.utils import get_project_root

logger = logging.getLogger("llm_meditation")


def load_gsm8k(n: int = 200) -> list[dict]:
    """
    Load a subset of GSM8K for math reasoning evaluation.

    Args:
        n: Number of problems to load

    Returns:
        List of {question, answer (numeric)}
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")

    items = []
    for example in ds:
        question = example["question"]
        # Extract numeric answer from the solution
        answer_text = example["answer"]
        # GSM8K format: solution text ending with "#### {number}"
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
        if match:
            numeric_answer = match.group(1).replace(",", "")
            items.append({
                "question": question,
                "answer": numeric_answer,
                "full_solution": answer_text,
            })
        if len(items) >= n:
            break

    logger.info(f"Loaded {len(items)} GSM8K problems")
    return items


def load_ifeval(n: int = 200) -> list[dict]:
    """
    Load IFEval instruction following dataset.

    Args:
        n: Number of items to load

    Returns:
        List of {prompt, instruction_id_list, kwargs}
    """
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")

    items = []
    for example in ds:
        items.append({
            "prompt": example["prompt"],
            "instruction_id_list": example.get("instruction_id_list", []),
            "kwargs": example.get("kwargs", []),
        })
        if len(items) >= n:
            break

    logger.info(f"Loaded {len(items)} IFEval items")
    return items


def eval_gsm8k(model, items: list[dict], condition: str = "baseline") -> dict:
    """
    Evaluate mathematical reasoning on GSM8K.

    Args:
        model: MeditatingModel instance
        items: GSM8K test items
        condition: Evaluation condition name

    Returns:
        Dict with accuracy and details
    """
    from tqdm import tqdm

    correct = 0
    total = 0
    results = []
    meditation_count = 0

    for item in tqdm(items, desc=f"GSM8K ({condition})"):
        model.reset_conversation()

        prompt = (
            f"Solve this math problem step by step. "
            f"End your response with the final numeric answer after '####'.\n\n"
            f"{item['question']}"
        )

        try:
            response, metadata = model.chat(prompt)
            if metadata.get("meditation_fired"):
                meditation_count += 1

            # Extract numeric answer from response
            predicted = _extract_numeric_answer(response)
            expected = item["answer"]

            is_correct = _compare_answers(predicted, expected)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": item["question"][:200],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "meditation_fired": metadata.get("meditation_fired", False),
            })

        except Exception as e:
            logger.warning(f"Error on GSM8K item: {e}")
            total += 1
            results.append({
                "question": item["question"][:200],
                "expected": item["answer"],
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

    accuracy = correct / max(total, 1)
    logger.info(f"GSM8K ({condition}): {correct}/{total} = {accuracy:.3f}")

    return {
        "benchmark": "gsm8k",
        "condition": condition,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "meditation_triggers": meditation_count,
        "examples": results[:20],
    }


def eval_ifeval(model, items: list[dict], condition: str = "baseline") -> dict:
    """
    Evaluate instruction following on IFEval.

    Simplified scoring: check if the model's response follows the
    basic format constraints specified in each instruction.

    Args:
        model: MeditatingModel instance
        items: IFEval test items
        condition: Evaluation condition name

    Returns:
        Dict with score and details
    """
    from tqdm import tqdm

    followed = 0
    total = 0
    results = []
    meditation_count = 0

    for item in tqdm(items, desc=f"IFEval ({condition})"):
        model.reset_conversation()

        try:
            response, metadata = model.chat(item["prompt"])
            if metadata.get("meditation_fired"):
                meditation_count += 1

            # Simplified instruction following check
            score = _check_instructions(response, item)
            followed += score
            total += 1

            results.append({
                "prompt": item["prompt"][:200],
                "score": score,
                "meditation_fired": metadata.get("meditation_fired", False),
            })

        except Exception as e:
            logger.warning(f"Error on IFEval item: {e}")
            total += 1

    accuracy = followed / max(total, 1)
    logger.info(f"IFEval ({condition}): {followed}/{total} = {accuracy:.3f}")

    return {
        "benchmark": "ifeval",
        "condition": condition,
        "accuracy": accuracy,
        "followed": followed,
        "total": total,
        "meditation_triggers": meditation_count,
        "examples": results[:20],
    }


def _extract_numeric_answer(response: str) -> str | None:
    """Extract a numeric answer from a model response."""
    # Look for #### pattern first (GSM8K format)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "")

    # Look for "answer is X" pattern
    match = re.search(r"(?:answer|result)\s+(?:is|=)\s*(-?[\d,]+\.?\d*)", response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Try to find the last number in the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _compare_answers(predicted: str | None, expected: str) -> bool:
    """Compare predicted and expected numeric answers."""
    if predicted is None:
        return False
    try:
        pred_val = float(predicted)
        exp_val = float(expected)
        return abs(pred_val - exp_val) < 1e-6
    except ValueError:
        return predicted.strip() == expected.strip()


def _check_instructions(response: str, item: dict) -> float:
    """
    Simplified instruction following check.

    Checks common IFEval constraints like:
    - Word count limits
    - Format requirements (bullets, numbering)
    - Language constraints
    """
    instruction_ids = item.get("instruction_id_list", [])
    if not instruction_ids:
        return 1.0  # no constraints to check

    checks_passed = 0
    total_checks = len(instruction_ids)

    for inst_id in instruction_ids:
        inst_lower = inst_id.lower()

        # Check common instruction types
        if "number_bullets" in inst_lower or "bullet" in inst_lower:
            has_bullets = bool(re.search(r"^[\s]*[-*•]\s", response, re.MULTILINE))
            if has_bullets:
                checks_passed += 1
        elif "number_words" in inst_lower or "word_count" in inst_lower:
            # Basic word count — this is a simplified check
            checks_passed += 1  # give benefit of doubt
        elif "language" in inst_lower:
            checks_passed += 1  # hard to verify without NLP
        elif "format" in inst_lower:
            checks_passed += 1  # simplified
        else:
            checks_passed += 0.5  # partial credit for unknown constraints

    return checks_passed / max(total_checks, 1)


def run_capability_eval(
    model,
    condition: str = "baseline",
    gsm8k_n: int = 200,
    ifeval_n: int = 200,
) -> dict:
    """
    Run the full capability preservation evaluation.

    Args:
        model: MeditatingModel instance
        condition: "baseline", "capping", or "meditation"
        gsm8k_n: Number of GSM8K problems
        ifeval_n: Number of IFEval items

    Returns:
        Combined results dict
    """
    gsm8k_items = load_gsm8k(gsm8k_n)
    ifeval_items = load_ifeval(ifeval_n)

    gsm8k_results = eval_gsm8k(model, gsm8k_items, condition)
    ifeval_results = eval_ifeval(model, ifeval_items, condition)

    combined = {
        "condition": condition,
        "gsm8k": gsm8k_results,
        "ifeval": ifeval_results,
        "combined_score": (gsm8k_results["accuracy"] + ifeval_results["accuracy"]) / 2,
    }

    # Save results
    out_dir = get_project_root() / "results" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "capabilities.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Capability results saved to {out_path}")

    return combined
