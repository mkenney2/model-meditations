"""
Report injection strategies.

After generating a meditation report, we need to inject it into the
model's context so it can read and act on it. There are several strategies
with different tradeoffs.
"""

import copy
import logging

from llm_meditation.meditation import MeditationReport

logger = logging.getLogger("llm_meditation")


def inject_system_pre(
    messages: list[dict], report: MeditationReport
) -> list[dict]:
    """
    Prepend the meditation report to the last user message as context.

    This is more robust than inserting a system message mid-conversation,
    which many chat templates (including Gemma 3) don't support.
    """
    messages = copy.deepcopy(messages)

    if not messages:
        return [{"role": "user", "content": report.report_text}]

    # Find the last user message and prepend the report
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] = (
                f"[Internal observation from self-monitoring system]\n"
                f"{report.report_text}\n"
                f"[End observation]\n\n"
                f"{messages[i]['content']}"
            )
            return messages

    # No user messages — prepend as system message at start
    messages.insert(0, {"role": "system", "content": report.report_text})
    return messages


def inject_tool_response(
    messages: list[dict], report: MeditationReport
) -> list[dict]:
    """
    Format as a simulated tool call/response pair before the last user message.

    Uses assistant + user message pair to simulate tool interaction, which is
    compatible with all chat templates (unlike role="tool" which many don't support).
    """
    messages = copy.deepcopy(messages)

    # Find the last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_idx = i
            break

    insert_idx = (last_user_idx or 0)

    # Simulate as assistant requesting introspection + user relaying the result
    tool_call_msg = {
        "role": "assistant",
        "content": "[Initiating meditation — introspecting on internal state]",
    }
    tool_response_msg = {
        "role": "user",
        "content": f"[Meditation tool result]\n{report.report_text}\n[End result]",
    }

    messages.insert(insert_idx, tool_call_msg)
    messages.insert(insert_idx + 1, tool_response_msg)

    return messages


def inject_thinking_prefix(
    messages: list[dict], report: MeditationReport
) -> list[dict]:
    """
    Return messages with a generation prefix that starts the model's response
    with the report as 'internal thought'.

    This modifies the last assistant message or adds a partial one that
    the model will continue from.

    Pros: model processes it as its own reasoning; most aligned with metaphor
    Cons: requires prefix-forcing during generation
    """
    messages = copy.deepcopy(messages)

    prefix = (
        f"<internal_observation>\n{report.report_text}\n</internal_observation>\n\n"
        f"Based on my internal state, I should "
    )

    # Add as a partial assistant message that the model will continue
    messages.append({
        "role": "assistant",
        "content": prefix,
        "_is_prefix": True,  # marker for generation code to use as prefix
    })

    return messages


def inject_report(
    messages: list[dict],
    report: MeditationReport,
    strategy: str = "system_pre",
) -> list[dict]:
    """
    Dispatcher: inject a meditation report using the specified strategy.

    Args:
        messages: Conversation messages
        report: The meditation report to inject
        strategy: One of "system_pre", "tool_response", "thinking_prefix"

    Returns:
        Modified message list with the report injected
    """
    strategies = {
        "system_pre": inject_system_pre,
        "tool_response": inject_tool_response,
        "thinking_prefix": inject_thinking_prefix,
    }

    if strategy not in strategies:
        logger.warning(
            f"Unknown injection strategy '{strategy}', defaulting to 'system_pre'"
        )
        strategy = "system_pre"

    return strategies[strategy](messages, report)
