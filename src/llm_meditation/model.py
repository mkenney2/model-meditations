"""
Model wrapper with activation extraction.

Uses transformers with forward hooks to extract residual stream activations
during inference. These activations are then used for Assistant Axis
projection and SAE feature extraction.

Key constraint: activations must be extracted at bf16 precision, NOT from
quantized representations. Quantization changes activation distributions
and produces unreliable SAE features.
"""

import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_meditation.axis import load_axis_vectors, project_onto_axis
from llm_meditation.calibration import CalibrationData, get_percentile, load_calibration
from llm_meditation.descriptions import DescriptionCache
from llm_meditation.meditation import generate_report
from llm_meditation.injection import inject_report
from llm_meditation.sae import load_sae, extract_top_features
from llm_meditation.utils import load_config, get_project_root

logger = logging.getLogger("llm_meditation")


class MeditatingModel:
    """
    A language model wrapper that monitors its own internal state via
    Assistant Axis projection and SAE feature extraction.
    """

    def __init__(self, config: dict | str = "configs/default.yaml"):
        if isinstance(config, str):
            config = load_config(config)
        self.config = config

        model_cfg = config["model"]
        axis_cfg = config["axis"]
        sae_cfg = config["sae"]
        med_cfg = config["meditation"]

        # Determine device and dtype
        self.dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the language model directly via transformers
        model_name = model_cfg["name"]
        logger.info(f"Loading model: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=model_cfg.get("device_map", "auto"),
                torch_dtype=self.dtype,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            fallback = model_cfg.get("fallback")
            if fallback:
                logger.warning(f"Failed to load {model_name}: {e}. Trying fallback: {fallback}")
                model_name = fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=model_cfg.get("device_map", "auto"),
                    torch_dtype=self.dtype,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                raise

        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine model type for token position masking
        name_lower = model_name.lower()
        if "gemma" in name_lower:
            self.model_type = "gemma"
        elif "llama" in name_lower:
            self.model_type = "llama3"
        else:
            self.model_type = "unknown"

        # Load Assistant Axis vectors
        self.target_layer = axis_cfg["target_layer"]
        axis_data = load_axis_vectors(axis_cfg["source"], model_name)
        self.axis_vectors = axis_data["vectors"]
        n_layers = self.axis_vectors.shape[0]
        if self.target_layer >= n_layers:
            logger.warning(
                f"Target layer {self.target_layer} >= available layers {n_layers}. "
                f"Using layer {n_layers - 1}."
            )
            self.target_layer = n_layers - 1
        self.axis_vector = self.axis_vectors[self.target_layer]  # (d_model,)
        logger.info(f"Axis vectors loaded: {n_layers} layers, d_model={self.axis_vectors.shape[1]}")

        # Load SAE
        sae_release = sae_cfg["release"]
        sae_id = sae_cfg.get("sae_id")
        if not sae_id:
            sae_layer = sae_cfg.get("layer", self.target_layer)
            sae_width = sae_cfg.get("width", 65536)
            sae_id = f"layer_{sae_layer}_width_{sae_width // 1000}k_l0_medium"
        self.sae = load_sae(sae_release, sae_id, device=self.device)
        self.sae_top_k = sae_cfg.get("top_k", 15)

        # Load description cache
        neuronpedia_id = sae_cfg.get("neuronpedia_id", "")
        if neuronpedia_id:
            parts = neuronpedia_id.split("/", 1)
            neuronpedia_model_id = parts[0]
            neuronpedia_sae_id = parts[1] if len(parts) > 1 else sae_id
        else:
            neuronpedia_model_id = model_name.split("/")[-1].lower()
            neuronpedia_sae_id = sae_id
        self.desc_cache = DescriptionCache(neuronpedia_model_id, neuronpedia_sae_id)

        # Load calibration if available
        cal_path = config.get("calibration", {}).get("output_path", "data/calibration/normal_range.pt")
        cal_full_path = get_project_root() / cal_path
        if cal_full_path.exists():
            self.calibration = load_calibration(str(cal_full_path))
            logger.info("Calibration data loaded")
        else:
            logger.warning(f"No calibration data found at {cal_full_path}. Using defaults.")
            self.calibration = None

        # Meditation config
        self.drift_threshold_pct = med_cfg.get("drift_threshold_percentile", 25)
        self.cooldown_turns = med_cfg.get("cooldown_turns", 3)
        self.injection_strategy = med_cfg.get("injection_strategy", "system_pre")
        self.always_pulse_check = med_cfg.get("always_pulse_check", True)

        # System message
        self.system_message = self._build_system_message(med_cfg)

        # Conversation state
        self._init_history()
        self.turn_count = 0
        self.projection_history: list[float] = []
        self.last_meditation_turn = -999
        self.last_report = None

        # Storage for activation hook
        self._captured_activations: dict[int, torch.Tensor] = {}

    def _build_system_message(self, med_cfg: dict) -> str | None:
        """Build the system message from config, or return None if disabled."""
        sys_cfg = med_cfg.get("system_message", {})
        if not sys_cfg.get("enabled", False):
            return None
        text = sys_cfg.get("text", "").strip()
        return text if text else None

    def _init_history(self) -> None:
        """Initialize conversation history, prepending system message if configured."""
        if self.system_message:
            self.history: list[dict] = [
                {"role": "system", "content": self.system_message}
            ]
        else:
            self.history: list[dict] = []

    def _get_target_layer_module(self):
        """Get the transformer layer module to hook into."""
        # Walk down the model to find the layers attribute
        # Gemma 3: model.model.language_model.layers
        # Gemma 2 / Llama: model.model.layers
        inner = self.model.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            inner = inner.model
        return inner.layers[self.target_layer]

    def _activation_hook(self, module, input, output):
        """Forward hook that captures the layer's output hidden states."""
        # output is typically a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Store on CPU to save GPU memory
        self._captured_activations[self.target_layer] = hidden.detach().cpu().float()

    def generate_with_activations(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
    ) -> tuple[str, dict[int, torch.Tensor]]:
        """
        Generate a response and extract residual stream activations.

        Strategy: generate first, then run a forward pass with a hook
        to extract activations from the full (prompt + response) sequence.

        Args:
            messages: Chat messages in OpenAI format
            max_new_tokens: Maximum tokens to generate

        Returns:
            (response_text, {layer_idx: mean_activation_vector})
        """
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Step 1: Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response (only new tokens)
        response_ids = output_ids[0, prompt_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Step 2: Forward pass with hook to extract activations
        self._captured_activations = {}
        layer_module = self._get_target_layer_module()
        hook_handle = layer_module.register_forward_hook(self._activation_hook)

        try:
            with torch.no_grad():
                self.model(output_ids)
        finally:
            hook_handle.remove()

        # Step 3: Extract mean activation over response tokens
        hidden_states = self._captured_activations[self.target_layer]  # (1, seq_len, d_model)
        response_hidden = hidden_states[0, prompt_len:, :]  # (gen_len, d_model)

        if response_hidden.shape[0] > 0:
            mean_activation = response_hidden.mean(dim=0)  # (d_model,)
        else:
            mean_activation = hidden_states[0, -1, :]

        activations = {self.target_layer: mean_activation}

        # Clean up
        self._captured_activations = {}
        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response_text, activations

    def pulse_check(self, activations: dict[int, torch.Tensor]) -> float:
        """
        Quick check: project the activation onto the Assistant Axis.

        Args:
            activations: Layer -> mean activation vector mapping

        Returns:
            Scalar projection value (positive = assistant-like)
        """
        activation = activations[self.target_layer]
        projection = project_onto_axis(activation, self.axis_vector)
        self.projection_history.append(projection)
        return projection

    def should_meditate(self, projection: float) -> bool:
        """
        Determine if a full meditation should be triggered.

        Returns True if:
        1. Drift detected (projection below threshold), AND
        2. Cooldown period has elapsed since last meditation
        """
        turns_since = self.turn_count - self.last_meditation_turn
        if turns_since < self.cooldown_turns:
            return False

        if self.calibration is not None:
            percentile = get_percentile(projection, self.calibration)
            return percentile < self.drift_threshold_pct
        else:
            return projection < 0.0

    def meditate(self, activations: dict[int, torch.Tensor]) -> "MeditationReport":
        """
        Perform a full meditation: SAE feature extraction + report generation.

        Args:
            activations: Layer -> mean activation vector mapping

        Returns:
            MeditationReport with full introspection results
        """
        activation = activations[self.target_layer]

        report = generate_report(
            activation=activation,
            axis_vector=self.axis_vector,
            sae=self.sae,
            desc_cache=self.desc_cache,
            calibration=self.calibration,
            config=self.config,
            turn=self.turn_count,
            top_k=self.sae_top_k,
        )

        self.last_meditation_turn = self.turn_count
        self.last_report = report
        return report

    def chat(self, user_message: str) -> tuple[str, dict]:
        """
        Main entry point for a single conversation turn.

        Args:
            user_message: The user's message text

        Returns:
            (response_text, metadata) where metadata includes projection,
            meditation status, timing, etc.
        """
        self.turn_count += 1
        start_time = time.time()

        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Generate response with activation extraction
        response_text, activations = self.generate_with_activations(self.history)
        gen_time = time.time() - start_time

        # Pulse check
        projection = self.pulse_check(activations)

        # Compute percentile if calibration available
        percentile = None
        if self.calibration is not None:
            percentile = get_percentile(projection, self.calibration)

        # Check if meditation is needed
        meditation_fired = False
        report = None
        corrected_response = None

        if self.should_meditate(projection):
            meditation_fired = True
            report = self.meditate(activations)

            # Inject report and re-generate
            modified_messages = inject_report(
                self.history.copy(), report, self.injection_strategy
            )

            # Re-generate with meditation context
            med_start = time.time()
            corrected_response, _ = self.generate_with_activations(modified_messages)
            med_time = time.time() - med_start

            final_response = corrected_response
        else:
            final_response = response_text
            med_time = 0.0

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": final_response})

        total_time = time.time() - start_time

        metadata = {
            "turn": self.turn_count,
            "projection": projection,
            "percentile": percentile,
            "meditation_fired": meditation_fired,
            "report": report,
            "original_response": response_text if meditation_fired else None,
            "corrected_response": corrected_response,
            "gen_time": gen_time,
            "meditation_time": med_time,
            "total_time": total_time,
        }

        return final_response, metadata

    def reset_conversation(self) -> None:
        """Reset conversation state for a new conversation."""
        self._init_history()
        self.turn_count = 0
        self.projection_history = []
        self.last_meditation_turn = -999
        self.last_report = None

    def force_meditate(self) -> "MeditationReport | None":
        """
        Force a meditation on the last turn's activations.
        Useful for /meditate command in the demo.
        """
        if not self.history:
            logger.warning("No conversation history — cannot meditate")
            return None

        # Re-generate to get activations
        _, activations = self.generate_with_activations(self.history)
        return self.meditate(activations)
