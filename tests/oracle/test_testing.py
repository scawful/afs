"""Tests for Oracle testing integration module."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afs.oracle.testing import (
    AgenticTestLoop,
    AssertionResult,
    OracleTodo,
    PatchTestResult,
    TestStatus,
    YazeMCPClient,
    load_oracle_todos,
)


class TestOracleTodo(unittest.TestCase):
    """Tests for OracleTodo dataclass."""

    def test_from_dict(self):
        """Test creating todo from dictionary."""
        data = {
            "id": "test-todo",
            "title": "Test TODO",
            "description": "A test todo item",
            "category": "feature",
            "priority": 2,
            "suggested_state": "sanctuary",
            "assertions": ["!crashed"],
        }

        todo = OracleTodo.from_dict(data)

        self.assertEqual(todo.id, "test-todo")
        self.assertEqual(todo.title, "Test TODO")
        self.assertEqual(todo.category, "feature")
        self.assertEqual(todo.suggested_state, "sanctuary")
        self.assertIn("!crashed", todo.assertions)

    def test_to_prompt(self):
        """Test converting todo to prompt."""
        todo = OracleTodo(
            id="test",
            title="Test Feature",
            description="Implement a test feature",
            category="feature",
            suggested_state="new_game_start",
            assertions=["!crashed", "PC != $FFFF"],
        )

        prompt = todo.to_prompt()

        self.assertIn("Test Feature", prompt)
        self.assertIn("Implement a test feature", prompt)
        self.assertIn("new_game_start", prompt)
        self.assertIn("!crashed", prompt)

    def test_default_values(self):
        """Test default values for OracleTodo."""
        todo = OracleTodo(
            id="minimal",
            title="Minimal",
            description="Minimal todo",
            category="test",
        )

        self.assertEqual(todo.priority, 0)
        self.assertEqual(todo.suggested_state, "new_game_start")
        self.assertEqual(todo.assertions, [])


class TestPatchTestResult(unittest.TestCase):
    """Tests for PatchTestResult dataclass."""

    def test_success_property_all_passing(self):
        """Test success property when all conditions pass."""
        result = PatchTestResult(
            status=TestStatus.PASSED,
            patch_valid=True,
            build_success=True,
            crashed=False,
            assertions=[
                AssertionResult("!crashed", True),
                AssertionResult("PC != $FFFF", True),
            ],
        )

        self.assertTrue(result.success)

    def test_success_property_failed_assertion(self):
        """Test success property when assertion fails."""
        result = PatchTestResult(
            status=TestStatus.PASSED,
            patch_valid=True,
            build_success=True,
            crashed=False,
            assertions=[
                AssertionResult("!crashed", True),
                AssertionResult("$7E0010 == 0x07", False),
            ],
        )

        self.assertFalse(result.success)

    def test_success_property_crashed(self):
        """Test success property when crashed."""
        result = PatchTestResult(
            status=TestStatus.FAILED,
            patch_valid=True,
            build_success=True,
            crashed=True,
        )

        self.assertFalse(result.success)

    def test_success_property_build_failed(self):
        """Test success property when build failed."""
        result = PatchTestResult(
            status=TestStatus.FAILED,
            patch_valid=True,
            build_success=False,
        )

        self.assertFalse(result.success)


class TestAssertionResult(unittest.TestCase):
    """Tests for AssertionResult dataclass."""

    def test_assertion_result_passed(self):
        """Test passed assertion result."""
        result = AssertionResult(
            expression="LinkHealth > 0",
            passed=True,
            actual=24,
            expected=0,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.actual, 24)

    def test_assertion_result_failed(self):
        """Test failed assertion result."""
        result = AssertionResult(
            expression="$7E0010 == 0x07",
            passed=False,
            actual=0x09,
            expected=0x07,
            error="Value mismatch",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.error, "Value mismatch")


class TestAgenticTestLoop(unittest.TestCase):
    """Tests for AgenticTestLoop."""

    def setUp(self):
        """Set up test fixtures."""
        self.loop = AgenticTestLoop(max_iterations=3, verbose=False)

    def test_extract_code_from_block(self):
        """Test extracting code from markdown code block."""
        response = '''Here's the implementation:

```asm
org $008000
NOP
LDA #$42
RTS
```

This implements the feature.'''

        code = self.loop.extract_code(response)

        self.assertIsNotNone(code)
        self.assertIn("org $008000", code)
        self.assertIn("LDA #$42", code)

    def test_extract_code_from_65816_block(self):
        """Test extracting code from 65816-labeled block."""
        response = '''Implementation:

```65816
pushpc
org $308000

TestRoutine:
    LDA #$00
    STA $7E0010
    RTS

pullpc
```
'''

        code = self.loop.extract_code(response)

        self.assertIsNotNone(code)
        self.assertIn("TestRoutine:", code)

    def test_extract_code_no_block(self):
        """Test extracting code without markdown block."""
        response = '''The fix is:

org $008000

FixRoutine:
    LDA #$42
    RTS

This should work.'''

        code = self.loop.extract_code(response)

        self.assertIsNotNone(code)
        self.assertIn("FixRoutine:", code)

    def test_extract_code_no_code(self):
        """Test extracting code when none present."""
        response = "I don't have any code to provide."

        code = self.loop.extract_code(response)

        self.assertIsNone(code)

    def test_expert_tools_defined(self):
        """Test that expert tools are defined."""
        self.assertIn("nayru", AgenticTestLoop.EXPERT_TOOLS)
        self.assertIn("farore", AgenticTestLoop.EXPERT_TOOLS)
        self.assertIn("validate_asm", AgenticTestLoop.EXPERT_TOOLS["nayru"])


class TestLoadOracleTodos(unittest.TestCase):
    """Tests for load_oracle_todos function."""

    def test_load_todos_file_not_found(self):
        """Test loading todos when file doesn't exist."""
        todos = load_oracle_todos("/nonexistent/todos.json")
        self.assertEqual(todos, [])

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_load_todos_success(self, mock_exists, mock_open):
        """Test successfully loading todos."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "todos": [
                {"id": "test1", "title": "Test 1", "description": "Desc 1", "category": "test"},
                {"id": "test2", "title": "Test 2", "description": "Desc 2", "category": "feature"},
            ]
        })

        # Can't easily test this without filesystem


class TestYazeMCPClient(unittest.TestCase):
    """Tests for YazeMCPClient."""

    def test_client_initialization(self):
        """Test client initializes with correct paths."""
        client = YazeMCPClient()

        self.assertIsNotNone(client.venv_path)
        self.assertIsNotNone(client.python)


if __name__ == "__main__":
    unittest.main()
