"""Tests for knowledge/entity_extractor.py entity extraction module."""

from __future__ import annotations

import pytest

from afs.knowledge.entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
    extract_entities,
    extract_with_stats,
)
from afs.knowledge.alttp_addresses import AddressCategory


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_entity_creation(self) -> None:
        entity = ExtractedEntity(
            address="$7EF36C",
            name="link_health",
            category=AddressCategory.LINK_STATE.value,
            confidence=1.0,
            line_number=1,
            is_known=True,
        )
        assert entity.address == "$7EF36C"
        assert entity.name == "link_health"
        assert entity.is_known is True


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_known_count(self) -> None:
        result = ExtractionResult()
        result.entities = [
            ExtractedEntity(
                address="$7EF36C",
                name="link_health",
                category="link_state",
                confidence=1.0,
                line_number=1,
                is_known=True,
            ),
            ExtractedEntity(
                address="$ABCD",
                name="addr_ABCD",
                category="unknown",
                confidence=0.0,
                line_number=2,
                is_known=False,
            ),
        ]
        assert result.known_count == 1

    def test_coverage(self) -> None:
        result = ExtractionResult()
        result.entities = [
            ExtractedEntity(
                address="$7EF36C",
                name="link_health",
                category="link_state",
                confidence=1.0,
                line_number=1,
                is_known=True,
            ),
            ExtractedEntity(
                address="$ABCD",
                name="addr_ABCD",
                category="unknown",
                confidence=0.0,
                line_number=2,
                is_known=False,
            ),
        ]
        assert result.coverage == pytest.approx(0.5)

    def test_coverage_empty(self) -> None:
        result = ExtractionResult()
        assert result.coverage == 0.0

    def test_entity_names(self) -> None:
        result = ExtractionResult()
        result.entities = [
            ExtractedEntity(
                address="$7EF36C",
                name="link_health",
                category="link_state",
                confidence=1.0,
                line_number=1,
                is_known=True,
            ),
            ExtractedEntity(
                address="$ABCD",
                name="addr_ABCD",
                category="unknown",
                confidence=0.0,
                line_number=2,
                is_known=False,
            ),
        ]
        names = result.entity_names()
        # Only known entities
        assert names == ["link_health"]


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor(include_hardware=True)

    def test_extract_known_address(self, extractor) -> None:
        code = """; Load Link's health
LDA $7EF36C
STA $00
RTS
"""
        result = extractor.extract(code)
        assert len(result.entities) > 0
        # Find the health entity
        health_entity = next(
            (e for e in result.entities if "health" in e.name.lower()),
            None,
        )
        assert health_entity is not None
        assert health_entity.is_known is True

    def test_extract_hardware_register(self, extractor) -> None:
        code = """LDA $2100  ; INIDISP
STA $2100
"""
        result = extractor.extract(code)
        # Should find INIDISP hardware register
        inidisp = next(
            (e for e in result.entities if e.name == "INIDISP"),
            None,
        )
        assert inidisp is not None
        assert inidisp.category == AddressCategory.HARDWARE.value

    def test_extract_unknown_address(self, extractor) -> None:
        code = """LDA $ABCD
STA $1234
"""
        result = extractor.extract(code)
        assert len(result.entities) > 0
        # Should have unknown addresses
        assert len(result.unknown_addresses) > 0

    def test_extract_labels(self, extractor) -> None:
        code = """MainLoop:
    LDA $00
    BNE MainLoop
SubRoutine:
    RTS
"""
        result = extractor.extract(code)
        assert "MainLoop" in result.labels
        assert "SubRoutine" in result.labels

    def test_skip_comment_lines(self, extractor) -> None:
        code = """; $7EF36C is Link's health
; This is just a comment
"""
        result = extractor.extract(code)
        # Should not extract addresses from comment-only lines
        assert len(result.entities) == 0

    def test_extract_different_address_formats(self, extractor) -> None:
        code = """LDA $7EF36C   ; Long address
LDA $F36C     ; Absolute
LDA $36       ; Direct page
"""
        result = extractor.extract(code)
        # Should extract all three address formats
        assert len(result.entities) >= 3

    def test_indexed_addressing(self, extractor) -> None:
        code = """LDA $0D00,X   ; Indexed by X
LDA $1000,Y   ; Indexed by Y
"""
        result = extractor.extract(code)
        # Should extract addresses even with indexing
        assert len(result.entities) >= 2

    def test_stats_populated(self, extractor) -> None:
        code = """LDA $7EF36C
STA $00
"""
        result = extractor.extract(code)
        assert "total_entities" in result.stats
        assert "known_entities" in result.stats
        assert "coverage" in result.stats
        assert "labels" in result.stats


class TestValidateEntities:
    """Tests for entity validation."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_validate_read_write_consistency(self, extractor) -> None:
        # This test checks that validation can detect read/write mismatches
        code = """LDA $7EF36C   ; Reading health is valid
"""
        result = extractor.extract(code)
        validation = extractor.validate_entities(result.entities, code)
        # Should validate without issues (reading from readable address)
        assert validation["is_valid"] is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_entities(self) -> None:
        code = """LDA $7EF36C
STA $00
"""
        names = extract_entities(code)
        assert isinstance(names, list)
        # Should return only known entity names
        assert all(isinstance(n, str) for n in names)

    def test_extract_with_stats(self) -> None:
        code = """LDA $7EF36C
LDA $2100
"""
        names, stats = extract_with_stats(code)
        assert isinstance(names, list)
        assert isinstance(stats, dict)
        assert "total_entities" in stats
        assert "coverage" in stats


class TestAddressResolution:
    """Tests for address resolution logic."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_long_address_bank_parsing(self, extractor) -> None:
        # Test that long addresses (6 hex digits) parse bank correctly
        code = """LDA $7EF36C"""
        result = extractor.extract(code)
        entity = result.entities[0]
        assert entity.address == "$7EF36C"

    def test_absolute_address_wram_assumption(self, extractor) -> None:
        # Test that absolute addresses in WRAM range get bank 7E assumption
        code = """LDA $0000"""  # Should be treated as $7E0000
        result = extractor.extract(code)
        # The extractor should find or create an entity for this
        assert len(result.entities) > 0

    def test_direct_page_handling(self, extractor) -> None:
        # Test 2-digit direct page addresses
        code = """LDA $00
LDA $FF"""
        result = extractor.extract(code)
        assert len(result.entities) == 2
