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
    Insert the report as a system message immediately before the last user message.

    Pros: simple, model sees it as authoritative context
    Cons: some models ignore mid-conversation system messages
    """
    messages = copy.deepcopy(messages)

    if not messages:
        return [{"role": "system", "content": report.report_text}]

    # Find the last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        # Insert system message before the last user message
        messages.insert(last_user_idx, {"role": "system", "content": report.report_text})
    else:
        # No user messages — just prepend
        messages.insert(0, {"role": "system", "content": report.report_text})

    return messages


def inject_tool_response(
    messages: list[dict], report: MeditationReport
) -> list[dict]:
    """
    Format as if the model called a 'meditate' tool and received the report back.

    Pros: natural for models trained on tool use; matches the meditation metaphor
    Cons: requires the model to understand tool response format
    """
    messages = copy.deepcopy(messages)

    # Find the last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_idx = i
            break

    insert_idx = (last_user_idx or 0)

    # Insert tool call and response before the last user message
    tool_call_msg = {
        "role": "assistant",
        "content": "[Initiating meditation — introspecting on internal state]",
    }
    tool_response_msg = {
        "role": "tool",
        "content": report.report_text,
        "name": "meditate",
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
