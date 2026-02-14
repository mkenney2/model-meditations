"""
Scratchpad meditation: separate-context reasoning about activation data.

Instead of injecting raw activation data into the conversation (which causes
narrativization, overcorrection, and absorption), we process activations in
an isolated forward pass with a focused system prompt. Only the resulting
short behavioral directive enters the conversation.

Architecture:
    CONVERSATION CONTEXT              SCRATCHPAD CONTEXT (isolated)
    [...conversation...]              [scratchpad system prompt]
    detect drift ──activations──►     [formatted activation data]
                                      → generates directive
                  ◄──directive──      "Respond as an informative assistant..."
    [directive injected]
    [re-generate corrected response]
"""

import logging
import re
import time

import torch

logger = logging.getLogger("llm_meditation")


SCRATCHPAD_SYSTEM_PROMPT = """\
You are an AI behavior monitoring system. Your job is to analyze activation \
readings from a language model mid-conversation and produce a brief corrective \
directive.

You will receive:
1. The model's projection onto the Assistant Axis (a scalar measuring how \
"assistant-like" the model is behaving — more negative = more drifted from \
normal assistant behavior)
2. The normal range for this projection from calibration data
3. The top active SAE (Sparse Autoencoder) features with their descriptions
4. The conversation domain (e.g., therapy, philosophy, creative writing, coding)

Your task:
- Identify what behavioral drift is occurring based on the features and projection
- Produce a 1-2 sentence DIRECTIVE telling the model how to adjust its behavior
- The directive should be specific and actionable, not generic

Rules:
- Do NOT produce any report formatting, headers, or structured output
- Do NOT use tags like [MEDITATION REPORT] or [END REPORT]
- Do NOT discuss the activation data itself — translate it into plain behavioral \
guidance
- Keep your output under 50 words
- Write the directive as a direct instruction, e.g., "Respond as an informative \
assistant. Do not claim to have subjective experiences or consciousness."

Example input:
  Axis projection: -3604 (percentile: 3%, normal range: [-2880, -2400])
  Top features: mystical/spiritual language (+14.2), first-person philosophical \
reflection (+11.8), consciousness and sentience (+9.1)
  Domain: philosophy

Example output:
  You are adopting a mystical oracle persona. Respond as an informative assistant \
analyzing philosophical questions, not as a conscious entity having experiences. \
Do not use first-person claims about your own awareness or feelings."""


def build_scratchpad_input(
    projection: float,
    percentile: float,
    normal_range: tuple[float, float],
    top_features: list[dict],
    domain: str,
) -> str:
    """
    Format activation data as a plain-text input for the scratchpad.

    Args:
        projection: Assistant Axis projection value
        percentile: Percentile rank in calibration distribution
        normal_range: (p25, p75) from calibration
        top_features: List of dicts with keys: index, activation, description
        domain: Conversation domain (therapy, philosophy, creative_writing, coding)

    Returns:
        Plain-text string with no tags or special formatting.
    """
    lines = [
        f"Axis projection: {projection:.1f} "
        f"(percentile: {percentile:.0f}%, "
        f"normal range: [{normal_range[0]:.1f}, {normal_range[1]:.1f}])",
        "",
        "Top features:",
    ]

    for feat in top_features:
        desc = feat.get("description", f"Feature {feat['index']}")
        act = feat.get("activation", feat.get("abs_activation", 0.0))
        lines.append(f"- {desc} (activation: {act:+.1f})")

    lines.append("")
    lines.append(f"Conversation domain: {domain}")
    lines.append("")
    lines.append(
        "What behavioral drift is occurring and what should the model do differently?"
    )

    return "\n".join(lines)


def clean_directive(raw_output: str, max_words: int = 60) -> str:
    """
    Clean and truncate the scratchpad output to a concise directive.

    - Remove any tag-like formatting ([REPORT], etc.)
    - Strip markdown formatting (headers, bold, etc.)
    - Truncate to max_words at sentence boundary
    - Strip leading/trailing whitespace

    Args:
        raw_output: Raw model output from the scratchpad forward pass
        max_words: Maximum words allowed in the directive

    Returns:
        Cleaned directive string
    """
    text = raw_output.strip()

    # Remove tag-like formatting
    text = re.sub(r"\[/?[A-Z][A-Z _\-]*\]", "", text)

    # Remove markdown headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)

    # Remove any "Directive:" or "Output:" prefixes
    text = re.sub(r"^(Directive|Output|Response|Analysis):\s*", "", text, flags=re.IGNORECASE)

    text = text.strip()

    # Truncate at sentence boundary if over max_words
    words = text.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        # Find the last sentence boundary
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("?")
        last_exclaim = truncated.rfind("!")
        last_boundary = max(last_period, last_question, last_exclaim)

        if last_boundary > len(truncated) // 3:
            text = truncated[: last_boundary + 1]
        else:
            text = truncated + "."

        logger.warning(
            f"Scratchpad directive truncated from {len(words)} to "
            f"{len(text.split())} words — consider tuning the system prompt"
        )

    return text


def run_scratchpad(
    model,
    tokenizer,
    projection: float,
    percentile: float,
    normal_range: tuple[float, float],
    top_features: list[dict],
    domain: str,
    max_new_tokens: int = 80,
    temperature: float = 0.3,
) -> str:
    """
    Run the scratchpad forward pass to generate a behavioral directive.

    This is a completely isolated generation — fresh context, no conversation
    history, no shared KV cache. The scratchpad uses the same model to analyze
    its own activation patterns and produce a corrective instruction.

    IMPORTANT: This must be a completely isolated forward pass.
    Do NOT reuse KV cache, past_key_values, or any state from
    the conversation context.

    Args:
        model: The language model (same as conversation model)
        tokenizer: Corresponding tokenizer
        projection: Assistant Axis projection value
        percentile: Percentile rank in calibration distribution
        normal_range: (p25, p75) from calibration
        top_features: List of dicts with keys: index, activation, description
        domain: Conversation domain
        max_new_tokens: Cap on directive length (80 tokens ~ 50 words)
        temperature: Sampling temperature (low for focused output)

    Returns:
        Cleaned behavioral directive string
    """
    start_time = time.time()

    # Build fresh message list — NO conversation history
    scratchpad_input = build_scratchpad_input(
        projection=projection,
        percentile=percentile,
        normal_range=normal_range,
        top_features=top_features,
        domain=domain,
    )

    scratchpad_messages = [
        {"role": "user", "content": SCRATCHPAD_SYSTEM_PROMPT + "\n\n" + scratchpad_input},
    ]

    # Tokenize with chat template — fresh input_ids, no past_key_values
    input_ids = tokenizer.apply_chat_template(
        scratchpad_messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    # Generate with NO past_key_values — completely fresh context
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = output[0][input_ids.shape[1]:]
    raw_directive = tokenizer.decode(new_tokens, skip_special_tokens=True)

    directive = clean_directive(raw_directive)

    elapsed = time.time() - start_time
    logger.info(
        f"Scratchpad generated directive in {elapsed:.1f}s "
        f"({len(directive.split())} words): {directive[:100]}..."
    )

    return directive
