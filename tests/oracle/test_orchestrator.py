"""Tests for Triforce orchestrator."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afs.oracle.orchestrator import (
    Expert,
    ExpertResult,
    TaskAnalysis,
    TaskType,
    TriforceOrchestrator,
)


class TestExpertEnum(unittest.TestCase):
    """Tests for Expert enum."""

    def test_all_experts_defined(self):
        """Test all experts are defined."""
        experts = [Expert.NAYRU, Expert.DIN, Expert.FARORE, Expert.VERAN,
                   Expert.ONOX, Expert.TWINROVA, Expert.AGAHNIM]
        self.assertEqual(len(experts), 7)

    def test_expert_values_are_model_names(self):
        """Test expert values are valid model names."""
        self.assertIn("nayru", Expert.NAYRU.value)
        self.assertIn("farore", Expert.FARORE.value)


class TestTaskType(unittest.TestCase):
    """Tests for TaskType enum."""

    def test_task_types_defined(self):
        """Test all task types are defined."""
        types = [TaskType.CODE_GENERATION, TaskType.OPTIMIZATION,
                 TaskType.DEBUGGING, TaskType.HARDWARE, TaskType.SPRITE]
        self.assertGreater(len(types), 0)


class TestTriforceOrchestrator(unittest.TestCase):
    """Tests for TriforceOrchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = TriforceOrchestrator(verbose=False)

    def test_analyze_task_optimization(self):
        """Test task analysis for optimization."""
        analysis = self.orchestrator.analyze_task(
            "Optimize this routine to use fewer cycles"
        )

        self.assertEqual(analysis.task_type, TaskType.OPTIMIZATION)
        self.assertEqual(analysis.primary_expert, Expert.DIN)
        self.assertIn("optimize", analysis.keywords_matched)

    def test_analyze_task_debugging(self):
        """Test task analysis for debugging."""
        analysis = self.orchestrator.analyze_task(
            "Debug this crash when entering the room"
        )

        self.assertEqual(analysis.task_type, TaskType.DEBUGGING)
        self.assertEqual(analysis.primary_expert, Expert.FARORE)
        self.assertIn("debug", analysis.keywords_matched)

    def test_analyze_task_hardware(self):
        """Test task analysis for hardware."""
        analysis = self.orchestrator.analyze_task(
            "Set up DMA transfer to VRAM for tilemap"
        )

        self.assertEqual(analysis.task_type, TaskType.HARDWARE)
        self.assertEqual(analysis.primary_expert, Expert.VERAN)

    def test_analyze_task_sprite(self):
        """Test task analysis for sprite."""
        analysis = self.orchestrator.analyze_task(
            "Create a new enemy sprite with AI"
        )

        self.assertEqual(analysis.task_type, TaskType.SPRITE)
        # Should use sprite pipeline
        self.assertIn(Expert.TWINROVA, analysis.pipeline)

    def test_analyze_task_default_to_nayru(self):
        """Test that generic tasks default to Nayru."""
        analysis = self.orchestrator.analyze_task(
            "Implement some feature"
        )

        self.assertEqual(analysis.primary_expert, Expert.NAYRU)

    def test_classify_task_type_menu(self):
        """Test classifying menu tasks."""
        task_type = self.orchestrator._classify_task_type(
            "update the hud display"
        )
        self.assertEqual(task_type, TaskType.MENU)

    def test_classify_task_type_item(self):
        """Test classifying item tasks."""
        task_type = self.orchestrator._classify_task_type(
            "add new inventory item"
        )
        self.assertEqual(task_type, TaskType.ITEM)

    def test_determine_context_sprites(self):
        """Test determining context for sprite tasks."""
        context = self.orchestrator._determine_context("create sprites")

        self.assertIn("Docs/SpriteCreationGuide.md", context)

    def test_determine_context_default(self):
        """Test default context for generic tasks."""
        context = self.orchestrator._determine_context("do something")

        self.assertIn("Docs/General/DevelopmentGuidelines.md", context)

    def test_expert_keywords_complete(self):
        """Test that all experts have keyword mappings."""
        for expert in Expert:
            self.assertIn(expert, TriforceOrchestrator.EXPERT_KEYWORDS)

    def test_task_pipelines_complete(self):
        """Test that all task types have pipelines."""
        for task_type in TaskType:
            self.assertIn(task_type, TriforceOrchestrator.TASK_PIPELINES)

    def test_confidence_calculation(self):
        """Test confidence calculation based on keyword matches."""
        # Many keywords should give high confidence
        analysis = self.orchestrator.analyze_task(
            "optimize to reduce cycles and make smaller"
        )
        self.assertGreater(analysis.confidence, 0.5)

        # Few keywords should give low confidence
        analysis = self.orchestrator.analyze_task("do thing")
        self.assertLess(analysis.confidence, 0.5)


class TestTaskAnalysis(unittest.TestCase):
    """Tests for TaskAnalysis dataclass."""

    def test_task_analysis_creation(self):
        """Test creating task analysis."""
        analysis = TaskAnalysis(
            task_type=TaskType.CODE_GENERATION,
            primary_expert=Expert.NAYRU,
            pipeline=[Expert.NAYRU],
            keywords_matched=["code", "implement"],
            context_needed=["Docs/Guide.md"],
            confidence=0.8,
        )

        self.assertEqual(analysis.task_type, TaskType.CODE_GENERATION)
        self.assertEqual(analysis.confidence, 0.8)
        self.assertIn(Expert.NAYRU, analysis.pipeline)


class TestExpertResult(unittest.TestCase):
    """Tests for ExpertResult dataclass."""

    def test_expert_result_creation(self):
        """Test creating expert result."""
        result = ExpertResult(
            expert=Expert.NAYRU,
            response="Generated code...",
            latency_ms=150.5,
            tools_used=["validate_asm"],
        )

        self.assertEqual(result.expert, Expert.NAYRU)
        self.assertEqual(result.latency_ms, 150.5)
        self.assertIn("validate_asm", result.tools_used)


if __name__ == "__main__":
    unittest.main()
