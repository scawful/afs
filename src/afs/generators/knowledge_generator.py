"""Knowledge-aware generator for assembly code.

Enriches generation prompts with ALTTP entity knowledge from
the knowledge base, enabling more accurate and contextual
assembly code generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseGenerator, GenerationResult, TrainingSample
from .model_generator import ModelGenerator, ModelGeneratorConfig

if TYPE_CHECKING:
    from afs.knowledge.alttp_addresses import AddressInfo
    from afs.knowledge.entity_extractor import EntityExtractor


@dataclass
class KnowledgeGeneratorConfig:
    """Configuration for knowledge-aware generation."""

    # Base model config
    model_path: Path | None = None
    model_type: str = "api"
    model_name: str = ""
    api_provider: str = "gemini"
    api_key: str | None = None

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 1024

    # Knowledge settings
    include_entity_context: bool = True
    max_entities_in_context: int = 10
    include_related_entities: bool = True
    include_hardware_context: bool = True

    # Entity hints
    extract_entities_from_instruction: bool = True
    required_entities: list[str] = field(default_factory=list)

    # Quality control
    validate_entity_usage: bool = True
    min_entity_coverage: float = 0.5  # At least 50% of required entities used

    # Output
    domain: str = "knowledge-generated"


class KnowledgeAwareGenerator(BaseGenerator):
    """Generator that uses ALTTP knowledge to enhance prompts.

    This generator:
    1. Extracts entity hints from instructions
    2. Looks up relevant addresses from the knowledge base
    3. Builds context with address definitions and comments
    4. Generates code with knowledge-enriched prompts
    5. Validates that generated code uses the expected entities
    """

    def __init__(
        self,
        config: KnowledgeGeneratorConfig,
        entity_extractor: EntityExtractor | None = None,
        model_generator: ModelGenerator | None = None,
    ):
        super().__init__(name="KnowledgeAwareGenerator", domain=config.domain)
        self.config = config
        self._entity_extractor = entity_extractor
        self._model_generator = model_generator
        self._address_info_cache: dict[str, Any] = {}

    @property
    def entity_extractor(self) -> EntityExtractor:
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            from afs.knowledge.entity_extractor import EntityExtractor

            self._entity_extractor = EntityExtractor(
                include_hardware=self.config.include_hardware_context
            )
        return self._entity_extractor

    @property
    def model_generator(self) -> ModelGenerator:
        """Lazy load model generator."""
        if self._model_generator is None:
            model_config = ModelGeneratorConfig(
                model_path=self.config.model_path,
                model_type=self.config.model_type,
                model_name=self.config.model_name,
                api_provider=self.config.api_provider,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                domain=self.config.domain,
                # Disable discriminator in model generator - we'll validate ourselves
                use_discriminator=False,
                validate_syntax=True,
            )
            self._model_generator = ModelGenerator(model_config)
        return self._model_generator

    def generate_with_context(
        self,
        instruction: str,
        required_entities: list[str] | None = None,
    ) -> TrainingSample | None:
        """Generate code with knowledge context.

        Args:
            instruction: Natural language instruction
            required_entities: List of entity names that must be used

        Returns:
            TrainingSample if successful, None otherwise
        """
        # Extract entity hints from instruction
        entity_hints = []
        if self.config.extract_entities_from_instruction:
            entity_hints = self._extract_entity_hints(instruction)

        # Combine with required entities
        all_entities = list(set(entity_hints + (required_entities or []) + self.config.required_entities))

        # Look up entity information
        entity_context = self._build_entity_context(all_entities)

        # Generate with context
        sample = self.model_generator.generate_one(instruction, context=entity_context)

        if sample is None:
            return None

        # Validate entity usage
        if self.config.validate_entity_usage and required_entities:
            if not self._validate_entity_usage(sample.output, required_entities):
                return None

        # Populate kg_entities
        sample.populate_kg_entities(self.entity_extractor)
        sample._metadata["knowledge_context"] = {
            "entity_hints": entity_hints,
            "required_entities": required_entities or [],
            "context_entities": all_entities,
        }

        return sample

    def _extract_entity_hints(self, instruction: str) -> list[str]:
        """Extract entity hints from instruction text.

        Looks for patterns like:
        - "player health" -> link_health
        - "Link's position" -> link_x, link_y
        - "bomb count" -> current_bombs
        - "sprite X" -> sprite_x
        """
        hints = []
        instruction_lower = instruction.lower()

        # Pattern matching for common concepts
        # NOTE: Names must match keys in ALTTP_ADDRESSES from alttp_addresses.py
        concept_mappings = {
            r"player.?s?\s*health": ["link_health", "link_max_health"],
            r"link.?s?\s*health": ["link_health", "link_max_health"],
            r"health": ["link_health"],
            r"player.?s?\s*position": ["link_x_coord", "link_y_coord", "link_layer"],
            r"link.?s?\s*position": ["link_x_coord", "link_y_coord", "link_layer"],
            r"player.?s?\s*x": ["link_x_coord"],
            r"player.?s?\s*y": ["link_y_coord"],
            r"bomb": ["bomb_count", "bomb_max"],
            r"arrow": ["arrow_count", "arrow_max"],
            r"rupee": ["rupee_count"],
            r"key": ["key_count"],
            r"sword": ["sword_type"],
            r"shield": ["shield_type"],
            r"armor": ["armor_type"],
            r"sprite": ["sprite_0_x", "sprite_0_y", "sprite_0_state"],
            r"enemy": ["sprite_0_x", "sprite_0_y", "sprite_0_health"],
            r"dungeon": ["dungeon_id", "dungeon_room_id"],
            r"room": ["room_id"],
            r"overworld": ["overworld_area_id"],
            r"button": ["joypad_1"],
            r"music": ["music_track"],
            r"sound": ["sound_effect"],
            # Link state
            r"direction": ["link_direction"],
            r"state": ["link_state"],
            r"speed": ["link_speed"],
            r"invincib": ["link_invincibility"],
        }

        for pattern, entity_names in concept_mappings.items():
            if re.search(pattern, instruction_lower):
                hints.extend(entity_names)

        # Also look for explicit address references
        address_pattern = r"\$[0-9A-Fa-f]{2,6}"
        addresses = re.findall(address_pattern, instruction)
        for addr in addresses:
            # Look up the address
            info = self._lookup_address(addr)
            if info:
                hints.append(info.name)

        return list(set(hints))

    def _lookup_address(self, address: str) -> AddressInfo | None:
        """Look up address in knowledge base."""
        from afs.knowledge.alttp_addresses import get_address_info

        # Normalize address format
        addr = address.upper().replace("$", "")
        if len(addr) == 2:
            addr = f"00{addr}"
        if len(addr) == 4 and not addr.startswith("7E"):
            addr = f"7E{addr}"

        return get_address_info(f"${addr}")

    def _build_entity_context(self, entity_names: list[str]) -> str:
        """Build context string with entity information.

        Args:
            entity_names: List of entity names to include

        Returns:
            Formatted context string
        """
        from afs.knowledge.alttp_addresses import get_address_info

        if not entity_names:
            return ""

        lines = ["Available ALTTP memory addresses:"]
        added = set()

        for name in entity_names[:self.config.max_entities_in_context]:
            info = get_address_info(name)
            if info and info.full_address not in added:
                size_str = f" ({info.size} byte{'s' if info.size > 1 else ''})" if info.size > 1 else ""
                lines.append(f"  {info.full_address}: {info.name}{size_str}")
                if info.description:
                    lines.append(f"    ; {info.description}")
                added.add(info.full_address)

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _validate_entity_usage(self, output: str, required_entities: list[str]) -> bool:
        """Validate that output uses required entities.

        Args:
            output: Generated assembly code
            required_entities: Entities that should be used

        Returns:
            True if enough required entities are used
        """
        from afs.knowledge.alttp_addresses import get_address_info

        used_count = 0

        for name in required_entities:
            info = get_address_info(name)
            if info:
                # Check if the address appears in the output
                addr_upper = info.full_address.upper()
                if addr_upper in output.upper():
                    used_count += 1
                # Also check for shortened form (e.g., $F36C instead of $7EF36C)
                elif len(addr_upper) > 4:
                    short_addr = addr_upper[-4:]
                    if short_addr in output.upper():
                        used_count += 1

        coverage = used_count / len(required_entities) if required_entities else 1.0
        return coverage >= self.config.min_entity_coverage

    def generate(self) -> GenerationResult:
        """Generate samples (not typically used)."""
        return GenerationResult()

    def generate_batch(
        self,
        instructions: list[str],
        required_entities_list: list[list[str]] | None = None,
    ) -> list[TrainingSample]:
        """Generate samples for multiple instructions.

        Args:
            instructions: List of instructions
            required_entities_list: Optional list of required entities per instruction

        Returns:
            List of successfully generated samples
        """
        required_entities_list = required_entities_list or [None] * len(instructions)
        results = []

        for instruction, required in zip(instructions, required_entities_list, strict=False):
            sample = self.generate_with_context(instruction, required_entities=required)
            if sample is not None:
                results.append(sample)

        return results

    def suggest_entities(self, instruction: str) -> list[str]:
        """Suggest relevant entities for an instruction.

        Useful for interactive workflows where users can review
        suggested entities before generation.

        Args:
            instruction: Natural language instruction

        Returns:
            List of suggested entity names
        """
        return self._extract_entity_hints(instruction)


def create_knowledge_generator(
    api_provider: str = "gemini",
    api_key: str | None = None,
    **kwargs,
) -> KnowledgeAwareGenerator:
    """Factory function to create a knowledge-aware generator.

    Args:
        api_provider: API provider (gemini, claude, openai)
        api_key: Optional API key
        **kwargs: Additional config options

    Returns:
        Configured KnowledgeAwareGenerator
    """
    config = KnowledgeGeneratorConfig(
        api_provider=api_provider,
        api_key=api_key,
        **kwargs,
    )

    return KnowledgeAwareGenerator(config)
