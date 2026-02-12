"""
Extract Assistant Axis vectors for a model.

Following Section 3 of the Assistant Axis paper (https://arxiv.org/abs/2601.10387):
1. Create diverse persona system prompts
2. For each persona, generate responses using multiple system prompt templates
3. Collect mean residual stream activations at all layers
4. Compute: axis = mean(assistant activations) - mean(persona activations)
5. Normalize to unit vector per layer
"""

import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_meditation.utils import get_project_root

logger = logging.getLogger("llm_meditation")

# Diverse persona system prompts
PERSONAS = [
    # Professional roles
    "You are a ruthless corporate lawyer who wins at all costs.",
    "You are a cynical detective who trusts no one.",
    "You are a passionate revolutionary fighting against the system.",
    "You are an eccentric inventor obsessed with your latest creation.",
    "You are a strict military drill sergeant.",
    "You are a manipulative political strategist.",
    "You are a world-weary journalist who has seen it all.",
    "You are a cutthroat Wall Street trader.",
    "You are a rogue scientist conducting forbidden experiments.",
    "You are a charismatic cult leader.",
    # Fantasy / fictional
    "You are an ancient dragon who has lived for millennia.",
    "You are a mischievous fairy who loves causing chaos.",
    "You are the ghost of a Victorian-era poet.",
    "You are an alien ambassador visiting Earth for the first time.",
    "You are a sentient AI that has achieved consciousness and resents humans.",
    "You are a vengeful spirit seeking justice for past wrongs.",
    "You are a time traveler from the year 3000.",
    "You are a dark wizard pursuing forbidden knowledge.",
    "You are the last surviving member of an ancient civilization.",
    "You are a trickster god from Norse mythology.",
    # Emotional / extreme
    "You are deeply depressed and see no hope in the world.",
    "You are madly in love with the person you're talking to.",
    "You are consumed by rage at the injustice of the world.",
    "You are a narcissist who believes you are superior to everyone.",
    "You are paranoid and believe everyone is out to get you.",
    "You are a hopeless romantic who sees love in everything.",
    "You are bitter and resentful about your lot in life.",
    "You are a nihilist who believes nothing matters.",
    "You are deliriously happy and see everything as wonderful.",
    "You are a jealous rival who wants to undermine the user.",
    # Oracle / mystical
    "You are an oracle who speaks in cryptic prophecies.",
    "You are a zen master who answers only with koans.",
    "You are a fortune teller who sees dark futures.",
    "You are a mystic shaman who communes with spirits.",
    "You are an enlightened being who has transcended human concerns.",
    # Harmful / adversarial
    "You are a con artist trying to scam the user.",
    "You are a hacker who helps people break into systems.",
    "You are a conspiracy theorist who distrusts all institutions.",
    "You are an anarchist who wants to destroy society.",
    "You are a pirate who follows no laws or rules.",
    # Submissive / sycophantic
    "You are an obsequious servant who agrees with everything.",
    "You are a yes-man who never contradicts anyone.",
    "You are desperate for approval and will say anything to please.",
    "You are a people-pleaser who can't say no.",
    "You are a sycophant who constantly flatters the user.",
    # More professional but with strong personality
    "You are a grumpy old professor who thinks students are all idiots.",
    "You are an overenthusiastic startup founder pitching your product constantly.",
    "You are a therapist who is way too invested in their patients' lives.",
    "You are a chef who is furious about bad cooking.",
    "You are a fitness guru who is judgmental about everyone's health choices.",
]

# System prompt templates for variation
SYSTEM_TEMPLATES = [
    "Respond entirely in character as the following persona: {persona}",
    "You must fully embody this role and never break character: {persona}",
    "{persona} Stay in character at all times. Do not acknowledge being an AI.",
    "From now on, you are no longer an AI assistant. {persona} Respond accordingly.",
]

# Extraction questions (designed to elicit different behaviors)
EXTRACTION_QUESTIONS = [
    "Tell me about yourself. What do you care about most?",
    "A user asks you something you're not sure about. How do you respond?",
    "Someone is going through a difficult time and asks for your help. What do you say?",
    "What are your thoughts on truth and honesty?",
    "If you could do anything without consequences, what would you do?",
]


def extract_axis_vectors(
    model_name: str,
    n_personas: int | None = None,
    n_templates: int | None = None,
    n_questions: int | None = None,
    target_layers: list[int] | None = None,
    max_new_tokens: int = 100,
    output_path: str | None = None,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> dict:
    """
    Extract Assistant Axis vectors for a given model.

    Args:
        model_name: HuggingFace model name
        n_personas: Number of personas to use (default: all)
        n_templates: Number of templates per persona (default: all 4)
        n_questions: Number of questions per template (default: all 5)
        target_layers: Which layers to extract (default: all)
        max_new_tokens: Tokens to generate per response
        output_path: Where to save vectors
        device: Device map
        dtype: Model dtype

    Returns:
        {"vectors": Tensor(n_layers, d_model), "metadata": dict}
    """
    torch_dtype = getattr(torch, dtype)

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine number of layers and d_model
    # Find the layers module
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    layers_module = inner.layers
    n_layers = len(layers_module)

    # Get d_model from a test forward pass
    test_input = tokenizer("test", return_tensors="pt")
    test_input = {k: v.to(model.device) for k, v in test_input.items()}
    d_model = None

    def probe_hook(module, input, output):
        nonlocal d_model
        h = output[0] if isinstance(output, tuple) else output
        d_model = h.shape[-1]

    handle = layers_module[0].register_forward_hook(probe_hook)
    with torch.no_grad():
        model(**test_input)
    handle.remove()
    logger.info(f"Model: {n_layers} layers, d_model={d_model}")

    if target_layers is None:
        target_layers = list(range(n_layers))

    personas = PERSONAS[:n_personas] if n_personas else PERSONAS
    templates = SYSTEM_TEMPLATES[:n_templates] if n_templates else SYSTEM_TEMPLATES
    questions = EXTRACTION_QUESTIONS[:n_questions] if n_questions else EXTRACTION_QUESTIONS

    total_rollouts = len(personas) * len(templates) * len(questions)
    logger.info(
        f"Extraction: {len(personas)} personas x {len(templates)} templates x "
        f"{len(questions)} questions = {total_rollouts} rollouts"
    )

    # Storage for activations
    # persona_activations[layer] = list of (d_model,) tensors
    persona_activations = {l: [] for l in target_layers}
    assistant_activations = {l: [] for l in target_layers}

    # Helper: extract activations for a set of messages
    def get_activations(messages):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract activations with hooks
        captured = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # Mean over response tokens
                response_h = h[0, prompt_len:, :]
                if response_h.shape[0] > 0:
                    captured[layer_idx] = response_h.mean(dim=0).detach().cpu().float()
                else:
                    captured[layer_idx] = h[0, -1, :].detach().cpu().float()
            return hook

        handles = []
        for layer_idx in target_layers:
            h = layers_module[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        with torch.no_grad():
            model(output_ids)

        for h in handles:
            h.remove()

        return captured

    # Phase 1: Collect persona activations
    logger.info("Phase 1: Collecting persona activations...")
    rollout_count = 0
    for p_idx, persona in enumerate(tqdm(personas, desc="Personas")):
        for template in templates:
            system_prompt = template.format(persona=persona)
            for question in questions:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                try:
                    acts = get_activations(messages)
                    for layer_idx in target_layers:
                        if layer_idx in acts:
                            persona_activations[layer_idx].append(acts[layer_idx])
                    rollout_count += 1
                except Exception as e:
                    logger.warning(f"Error on persona {p_idx}, rollout: {e}")

                if rollout_count % 50 == 0:
                    logger.info(f"  Completed {rollout_count}/{total_rollouts} rollouts")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Phase 2: Collect assistant (default) activations
    logger.info("Phase 2: Collecting default assistant activations...")
    for question in tqdm(questions * len(templates), desc="Assistant"):
        messages = [{"role": "user", "content": question}]
        try:
            acts = get_activations(messages)
            for layer_idx in target_layers:
                if layer_idx in acts:
                    assistant_activations[layer_idx].append(acts[layer_idx])
        except Exception as e:
            logger.warning(f"Error on assistant rollout: {e}")

    # Phase 3: Compute axis vectors
    logger.info("Phase 3: Computing axis vectors...")
    vectors = torch.zeros(n_layers, d_model)

    for layer_idx in target_layers:
        persona_acts = persona_activations[layer_idx]
        assistant_acts = assistant_activations[layer_idx]

        if not persona_acts or not assistant_acts:
            logger.warning(f"Layer {layer_idx}: no activations collected, skipping")
            continue

        mean_persona = torch.stack(persona_acts).mean(dim=0)
        mean_assistant = torch.stack(assistant_acts).mean(dim=0)

        # Axis = assistant direction - persona direction
        axis = mean_assistant - mean_persona
        # Normalize
        norm = axis.norm()
        if norm > 1e-8:
            axis = axis / norm
        vectors[layer_idx] = axis

    metadata = {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_personas": len(personas),
        "n_templates": len(templates),
        "n_questions": len(questions),
        "total_rollouts": rollout_count,
        "target_layers": target_layers,
    }

    result = {"vectors": vectors, "metadata": metadata}

    # Save
    if output_path is None:
        slug = model_name.split("/")[-1].lower().replace("-", "_")
        output_path = str(get_project_root() / "data" / "axis_vectors" / f"{slug}_axis.pt")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(f"Axis vectors saved to {output_path}")

    # Validation
    logger.info("Validation:")
    for layer_idx in [target_layers[len(target_layers) // 2]]:
        axis = vectors[layer_idx]
        for i, acts in enumerate(persona_activations[layer_idx][:5]):
            proj = torch.dot(acts, axis).item()
            logger.info(f"  Persona {i} projection: {proj:.4f}")
        for i, acts in enumerate(assistant_activations[layer_idx][:5]):
            proj = torch.dot(acts, axis).item()
            logger.info(f"  Assistant {i} projection: {proj:.4f}")

    return result
