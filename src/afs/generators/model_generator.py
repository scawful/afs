"""Model-based generator for assembly code.

Uses trained language models to generate ALTTP assembly code from
natural language instructions, with quality filtering via discriminator
and syntax validation.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from .base import BaseGenerator, GenerationResult, TrainingSample

if TYPE_CHECKING:
    from afs.discriminator.electra import ASMElectra
    from afs.generators.asar_validator import AsarValidator
    from afs.training.scoring import QualityScorer

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""

    MLX = "mlx"  # Apple MLX format
    HUGGINGFACE = "huggingface"  # Transformers format
    LLAMA_CPP = "llama_cpp"  # GGUF format
    API = "api"  # External API (Gemini, Claude, etc.)


def _normalize_model_type(value: str | "ModelType") -> str:
    if isinstance(value, ModelType):
        return value.value
    return str(value).strip().lower()


@dataclass
class ModelGeneratorConfig:
    """Configuration for model-based generation."""

    # Model settings
    model_path: Path | None = None
    model_type: str = ModelType.MLX.value
    model_name: str = ""  # For API or HuggingFace hub

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    repetition_penalty: float = 1.1

    # Quality control
    use_discriminator: bool = True
    discriminator_model_path: Path | None = None  # Path to ELECTRA discriminator
    min_quality_score: float = 0.6
    validate_syntax: bool = True
    max_retries: int = 3

    # API settings (for ModelType.API)
    api_provider: str = "gemini"  # gemini, claude, openai
    api_key: str | None = None

    # Output
    domain: str = "model-generated"

    def __post_init__(self) -> None:
        self.model_type = _normalize_model_type(self.model_type)


class ModelBackend(ABC):
    """Abstract backend for model inference."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class MLXBackend(ModelBackend):
    """MLX backend for Apple Silicon."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy load MLX model."""
        if self._model is not None:
            return

        try:
            from mlx_lm import load, generate

            self._model, self._tokenizer = load(str(self.model_path))
            self._generate_fn = generate
        except ImportError:
            raise ImportError("mlx-lm not installed. Install with: pip install mlx-lm")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        self._load_model()

        response = self._generate_fn(
            self._model,
            self._tokenizer,
            prompt=prompt,
            temp=temperature,
            max_tokens=max_tokens,
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        )

        return response

    def is_available(self) -> bool:
        try:
            import mlx
            return True
        except ImportError:
            return False


class LlamaCppBackend(ModelBackend):
    """llama.cpp backend for GGUF models."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None

    def _load_model(self) -> None:
        """Lazy load llama.cpp model."""
        if self._model is not None:
            return

        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=4096,
                n_gpu_layers=-1,  # Use all GPU layers
            )
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        self._load_model()

        response = self._model(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            repeat_penalty=kwargs.get("repetition_penalty", 1.1),
            top_p=kwargs.get("top_p", 0.9),
        )

        return response["choices"][0]["text"]

    def is_available(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False


class HuggingFaceBackend(ModelBackend):
    """Hugging Face Transformers backend."""

    def __init__(self, model_path: Path | str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy load HuggingFace model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        self._load_model()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                do_sample=True,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        return response[len(prompt):].strip()

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False


class APIBackend(ModelBackend):
    """API backend using existing LLM clients."""

    def __init__(self, provider: str, api_key: str | None = None, model: str | None = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load API client."""
        if self._client is not None:
            return self._client

        from afs.generators.cot.client import get_client

        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.model:
            kwargs["model"] = self.model

        self._client = get_client(self.provider, **kwargs)
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        client = self._get_client()
        system_prompt = kwargs.get("system_prompt", ASM_GENERATION_PROMPT)
        return client.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def is_available(self) -> bool:
        return True  # Depends on API key


BackendFactory = Callable[[ModelGeneratorConfig], ModelBackend]
_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a backend factory."""
    key = name.strip().lower()
    if not overwrite and key in _BACKEND_FACTORIES:
        raise ValueError(f"Backend already registered: {key}")
    _BACKEND_FACTORIES[key] = factory


def get_backend_factory(name: str) -> BackendFactory | None:
    return _BACKEND_FACTORIES.get(name.strip().lower())


def available_backends() -> list[str]:
    from ..plugins import load_enabled_plugins

    load_enabled_plugins()
    return sorted(_BACKEND_FACTORIES)


def _register_builtin_backends() -> None:
    if _BACKEND_FACTORIES:
        return

    def _mlx_factory(config: ModelGeneratorConfig) -> ModelBackend:
        if not config.model_path:
            raise ValueError("model_path required for MLX backend")
        return MLXBackend(config.model_path)

    def _llama_factory(config: ModelGeneratorConfig) -> ModelBackend:
        if not config.model_path:
            raise ValueError("model_path required for llama.cpp backend")
        return LlamaCppBackend(config.model_path)

    def _hf_factory(config: ModelGeneratorConfig) -> ModelBackend:
        model_id = config.model_name or config.model_path
        if not model_id:
            raise ValueError("model_path or model_name required for HuggingFace backend")
        return HuggingFaceBackend(model_id)

    def _api_factory(config: ModelGeneratorConfig) -> ModelBackend:
        return APIBackend(
            config.api_provider,
            api_key=config.api_key,
            model=config.model_name or None,
        )

    register_backend(ModelType.MLX.value, _mlx_factory)
    register_backend(ModelType.LLAMA_CPP.value, _llama_factory)
    register_backend(ModelType.HUGGINGFACE.value, _hf_factory)
    register_backend(ModelType.API.value, _api_factory)


# System prompt for ASM generation
ASM_GENERATION_PROMPT = """You are an expert 65816 assembly programmer specializing in SNES/Super Nintendo game development, particularly for The Legend of Zelda: A Link to the Past (ALTTP).

When given a natural language instruction, generate clean, well-commented 65816 assembly code that accomplishes the task. Follow these guidelines:

1. Use proper 65816 opcodes and addressing modes
2. Include clear comments explaining the code
3. Use meaningful labels when needed
4. Reference ALTTP WRAM addresses when appropriate (e.g., $7EF36C for Link's health)
5. Keep code efficient and idiomatic for SNES development

Generate ONLY the assembly code, without explanations or markdown formatting."""


class ModelGenerator(BaseGenerator):
    """Generator that uses trained models to create assembly code.

    Supports multiple backends (MLX, HuggingFace, llama.cpp, API)
    with quality filtering via discriminator and syntax validation.
    """

    def __init__(
        self,
        config: ModelGeneratorConfig,
        discriminator: "ASMElectra | None" = None,
        validator: "AsarValidator | None" = None,
        scorer: "QualityScorer | None" = None,
    ):
        super().__init__(name="ModelGenerator", domain=config.domain)
        self.config = config
        self._discriminator = discriminator
        self._validator = validator
        self._scorer = scorer
        self._backend: ModelBackend | None = None

    @property
    def backend(self) -> ModelBackend:
        """Lazy initialize model backend."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def _create_backend(self) -> ModelBackend:
        """Create appropriate backend based on config."""
        from ..plugins import load_enabled_plugins

        load_enabled_plugins()
        backend_type = _normalize_model_type(self.config.model_type)
        factory = get_backend_factory(backend_type)
        if factory is None:
            available = ", ".join(available_backends())
            raise ValueError(
                f"Unsupported model type: {self.config.model_type}. "
                f"Available: {available}"
            )
        return factory(self.config)

    @property
    def discriminator(self) -> "ASMElectra | None":
        """Lazy load discriminator."""
        if self._discriminator is None and self.config.use_discriminator:
            if self.config.discriminator_model_path:
                try:
                    from afs.discriminator.electra import ASMElectra
                    self._discriminator = ASMElectra(
                        model_path=self.config.discriminator_model_path
                    )
                    logger.info(
                        "Loaded discriminator from %s",
                        self.config.discriminator_model_path,
                    )
                except ImportError as e:
                    logger.warning(
                        "Could not import ASMElectra: %s. "
                        "Discriminator filtering disabled.",
                        e,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load discriminator from %s: %s. "
                        "Discriminator filtering disabled.",
                        self.config.discriminator_model_path,
                        e,
                    )
            else:
                logger.debug(
                    "No discriminator_model_path configured. "
                    "Discriminator filtering disabled."
                )
        return self._discriminator

    @property
    def validator(self) -> "AsarValidator":
        """Lazy load validator."""
        if self._validator is None:
            from afs.generators.asar_validator import AsarValidator, AsarValidatorConfig
            self._validator = AsarValidator(AsarValidatorConfig())
        return self._validator

    def generate_one(
        self,
        instruction: str,
        context: str = "",
    ) -> TrainingSample | None:
        """Generate a single sample with quality control.

        Args:
            instruction: Natural language instruction
            context: Optional context (e.g., related code, entity info)

        Returns:
            TrainingSample if quality passes, None otherwise
        """
        prompt = self._build_prompt(instruction, context)

        for attempt in range(self.config.max_retries):
            try:
                output = self.backend.generate(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                )

                # Clean output
                output = self._clean_output(output)

                if not output or len(output) < 10:
                    continue

                # Create sample
                sample = TrainingSample(
                    sample_id=str(uuid.uuid4()),
                    instruction=instruction,
                    output=output,
                    domain=self.config.domain,
                )

                # Quality check
                if self._passes_quality(sample):
                    sample._metadata["generation_attempt"] = attempt + 1
                    sample._metadata["model_type"] = self.config.model_type
                    return sample

            except Exception as e:
                sample_metadata = {"error": str(e), "attempt": attempt + 1}
                continue

        return None

    def _build_prompt(self, instruction: str, context: str = "") -> str:
        """Build generation prompt."""
        parts = []

        if context:
            parts.append(f"Context:\n{context}\n")

        parts.append(f"Instruction: {instruction}")
        parts.append("\nAssembly code:")

        return "\n".join(parts)

    def _clean_output(self, output: str) -> str:
        """Clean model output."""
        # Remove markdown code blocks
        if "```" in output:
            lines = output.split("\n")
            clean_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or not line.strip().startswith("```"):
                    clean_lines.append(line)
            output = "\n".join(clean_lines)

        # Remove common prefixes
        for prefix in ["Here's the assembly code:", "Here is the assembly:", "```asm", "```assembly"]:
            if output.strip().startswith(prefix):
                output = output[len(prefix):].strip()

        return output.strip()

    def _passes_quality(self, sample: TrainingSample) -> bool:
        """Check if sample passes quality thresholds."""
        # Syntax validation
        if self.config.validate_syntax:
            try:
                result = self.validator.validate_sample(sample)
                if not result.passed:
                    logger.debug(
                        "Sample %s failed syntax validation: %s",
                        sample.sample_id,
                        result.error_message,
                    )
                    return False
            except FileNotFoundError as e:
                # asar not installed - log once and continue without validation
                logger.warning("Syntax validation skipped: %s", e)
            except Exception as e:
                logger.debug(
                    "Syntax validation error for sample %s: %s",
                    sample.sample_id,
                    e,
                )

        # Discriminator check
        if self.discriminator is not None:
            try:
                score = self.discriminator.score(sample.output)
                sample._metadata["electra_score"] = score
                if score < self.config.min_quality_score:
                    logger.debug(
                        "Sample %s failed quality threshold: %.3f < %.3f",
                        sample.sample_id,
                        score,
                        self.config.min_quality_score,
                    )
                    return False
            except Exception as e:
                logger.debug(
                    "Discriminator scoring error for sample %s: %s",
                    sample.sample_id,
                    e,
                )

        return True

    def generate(self) -> GenerationResult:
        """Generate samples (not typically used - use generate_one or generate_batch)."""
        return GenerationResult()

    def generate_batch(
        self,
        instructions: list[str],
        contexts: list[str] | None = None,
    ) -> list[TrainingSample]:
        """Generate samples for multiple instructions.

        Args:
            instructions: List of instructions
            contexts: Optional list of contexts (same length as instructions)

        Returns:
            List of successfully generated samples
        """
        contexts = contexts or [""] * len(instructions)
        results = []

        for instruction, context in zip(instructions, contexts):
            sample = self.generate_one(instruction, context)
            if sample is not None:
                results.append(sample)

        return results


def create_generator(
    model_path: Path | str | None = None,
    model_type: str = "mlx",
    api_provider: str | None = None,
    **kwargs,
) -> ModelGenerator:
    """Factory function to create a model generator.

    Args:
        model_path: Path to model (for local models)
        model_type: Model type (mlx, huggingface, llama_cpp, api)
        api_provider: API provider (gemini, claude, openai) for API type
        **kwargs: Additional config options

    Returns:
        Configured ModelGenerator
    """
    config = ModelGeneratorConfig(
        model_path=Path(model_path) if model_path else None,
        model_type=model_type,
        api_provider=api_provider or "gemini",
        **kwargs,
    )

    return ModelGenerator(config)


_register_builtin_backends()
