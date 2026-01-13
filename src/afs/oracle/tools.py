"""Oracle of Secrets specific tools for ROM hacking.

Provides knowledge lookup and verification tools for the Triforce experts.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Knowledge base paths
KNOWLEDGE_BASE = Path.home() / ".context" / "knowledge"
ORACLE_PROJECT = Path.home() / "src" / "hobby" / "oracle-of-secrets"


@dataclass
class ToolDefinition:
    """Definition of a tool available to agents."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]


class OracleTools:
    """Collection of tools for Oracle of Secrets development."""

    def __init__(self):
        self.knowledge_cache: dict[str, str] = {}

    def lookup_snes_register(self, register: str) -> str:
        """Look up SNES register documentation.

        Args:
            register: Register address like "$2100" or "INIDISP"

        Returns:
            Documentation for the register or "not found"
        """
        # Load DMA registers knowledge
        dma_path = KNOWLEDGE_BASE / "snes" / "dma_registers.md"
        ppu_path = KNOWLEDGE_BASE / "snes" / "ppu_registers.md"

        result = []

        for path in [dma_path, ppu_path]:
            if path.exists():
                content = path.read_text()
                # Search for register
                reg_lower = register.lower().replace("$", "")
                if reg_lower in content.lower():
                    # Extract relevant section
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if reg_lower in line.lower():
                            # Get context around match
                            start = max(0, i - 2)
                            end = min(len(lines), i + 5)
                            result.append("\n".join(lines[start:end]))
                            break

        if result:
            return "\n\n".join(result)
        return f"Register {register} not found in knowledge base"

    def lookup_alttp_ram(self, address: str) -> str:
        """Look up ALTTP RAM address documentation.

        Args:
            address: RAM address like "$7E0022" or variable name like "LINKX"

        Returns:
            Documentation for the address or "not found"
        """
        ram_path = KNOWLEDGE_BASE / "alttp" / "ram_map.md"
        oracle_mem_path = ORACLE_PROJECT / "Docs" / "Core" / "MemoryMap.md"

        result = []

        for path in [ram_path, oracle_mem_path]:
            if path.exists():
                content = path.read_text()
                addr_lower = address.lower().replace("$", "")
                if addr_lower in content.lower():
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if addr_lower in line.lower():
                            start = max(0, i - 1)
                            end = min(len(lines), i + 3)
                            result.append("\n".join(lines[start:end]))
                            break

        if result:
            return "\n\n".join(result)
        return f"Address {address} not found in knowledge base"

    def lookup_oracle_docs(self, topic: str) -> str:
        """Look up Oracle of Secrets documentation.

        Args:
            topic: Topic to search for (e.g., "mask system", "time system", "sprites")

        Returns:
            Relevant documentation excerpt
        """
        docs_path = ORACLE_PROJECT / "Docs"

        if not docs_path.exists():
            return "Oracle docs not found"

        # Search through docs
        matches = []
        for doc in docs_path.rglob("*.md"):
            content = doc.read_text()
            if topic.lower() in content.lower():
                # Find relevant section
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if topic.lower() in line.lower():
                        start = max(0, i - 2)
                        end = min(len(lines), i + 10)
                        matches.append(f"**{doc.name}:**\n" + "\n".join(lines[start:end]))
                        break

        if matches:
            return "\n\n---\n\n".join(matches[:3])  # Limit to 3 matches
        return f"No documentation found for '{topic}'"

    def search_oracle_code(self, pattern: str, file_type: str = "asm") -> str:
        """Search Oracle of Secrets codebase for a pattern.

        Args:
            pattern: Regex pattern to search for
            file_type: File extension to search (default: asm)

        Returns:
            Matching code snippets with file locations
        """
        if not ORACLE_PROJECT.exists():
            return "Oracle project not found"

        results = []
        for asm_file in ORACLE_PROJECT.rglob(f"*.{file_type}"):
            try:
                content = asm_file.read_text()
                for i, line in enumerate(content.split("\n")):
                    if re.search(pattern, line, re.IGNORECASE):
                        rel_path = asm_file.relative_to(ORACLE_PROJECT)
                        results.append(f"{rel_path}:{i+1}: {line.strip()}")
            except Exception:
                continue

        if results:
            return "\n".join(results[:20])  # Limit results
        return f"No matches for pattern '{pattern}'"

    def check_memory_free(self, bank: str, start: str, end: str) -> str:
        """Check if a memory region is available.

        Args:
            bank: ROM bank (e.g., "2B")
            start: Start address within bank
            end: End address within bank

        Returns:
            Whether the region is free and any conflicts
        """
        # ROM bank allocations from Oracle docs
        bank_usage = {
            "20": "Expanded Music",
            "21": "ZScream Reserved", "22": "ZScream Reserved",
            "23": "ZScream Reserved", "24": "ZScream Reserved",
            "25": "ZScream Reserved", "26": "ZScream Reserved",
            "27": "ZScream Reserved",
            "28": "ZSCustomOverworld",
            "29": "ZScream Reserved", "2A": "ZScream Reserved",
            "2B": "Items",
            "2C": "Underworld/Dungeons",
            "2D": "Menu",
            "2E": "HUD",
            "2F": "Expanded Messages",
            "30": "Sprites", "31": "Sprites", "32": "Sprites",
            "33": "Moosh Form Gfx",
            "34": "Time System/Overlays",
            "35": "Deku Link Gfx",
            "36": "Zora Link Gfx",
            "37": "Bunny Link Gfx",
            "38": "Wolf Link Gfx",
            "39": "Minish Link Gfx",
            "3A": "Mask Routines",
            "3B": "GBC Link Gfx",
            "3C": "Unused",
            "3D": "ZS Tile16",
            "3E": "LW ZS Tile32",
            "3F": "DW ZS Tile32",
            "40": "LW World Map",
            "41": "DW World Map",
        }

        bank_upper = bank.upper()
        if bank_upper in bank_usage:
            usage = bank_usage[bank_upper]
            if usage == "Unused":
                return f"Bank ${bank_upper} is free for use"
            elif "Reserved" in usage:
                return f"Bank ${bank_upper} is {usage} - DO NOT USE"
            else:
                return f"Bank ${bank_upper} is used for: {usage}. Check for free space within the bank."
        else:
            return f"Bank ${bank_upper} not in allocation map"

    def get_sprite_table_format(self) -> str:
        """Get the Oracle sprite table entry format."""
        return """Oracle Sprite Table Entry Format:

Sprite Type Table: $0E20 (16 entries, 1 byte each)
Sprite State Table: $0DD0 (16 entries, 1 byte each)

Custom Sprite Definition (8 bytes):
  Byte 0: AI Type
  Byte 1: Max HP
  Byte 2: Damage
  Byte 3: Graphics Set ID
  Byte 4: Palette Index
  Byte 5: Flags (bit 7=invincible, bit 6=boss, bits 0-3=collision)
  Byte 6-7: Reserved

Standard sprite states:
  $00 = Inactive
  $01 = Active
  $02 = Dying
  $03 = Stunned
  $04-$0F = Custom states"""

    def validate_namespace(self, code: str) -> str:
        """Check code for namespace issues.

        Args:
            code: Assembly code to validate

        Returns:
            Validation result and suggestions
        """
        issues = []

        # Check for Oracle namespace usage
        if "namespace Oracle" in code:
            # Look for calls without Oracle_ prefix
            if re.search(r'\bJSL\s+[A-Za-z_]+\b', code):
                calls = re.findall(r'JSL\s+([A-Za-z_]+)', code)
                for call in calls:
                    if not call.startswith("Oracle_") and not call.startswith("$"):
                        issues.append(f"Call to '{call}' may need Oracle_ prefix")

        # Check for pushpc/pullpc balance
        push_count = code.count("pushpc")
        pull_count = code.count("pullpc")
        if push_count != pull_count:
            issues.append(f"Unbalanced pushpc/pullpc: {push_count} push, {pull_count} pull")

        # Check for missing bank specifiers
        if re.search(r'\borg\s+\$[0-9A-Fa-f]{4}\b', code):
            # org with only 4 hex digits (no bank)
            issues.append("org directive may be missing bank specifier (use $XX8000 format)")

        if issues:
            return "Issues found:\n- " + "\n- ".join(issues)
        return "No namespace issues detected"


# Create tool definitions for agent use
def _create_tool_defs(tools: OracleTools) -> dict[str, ToolDefinition]:
    """Create tool definitions from OracleTools instance."""
    return {
        "lookup_snes_register": ToolDefinition(
            name="lookup_snes_register",
            description="Look up documentation for a SNES hardware register",
            parameters={
                "type": "object",
                "properties": {
                    "register": {
                        "type": "string",
                        "description": "Register address (e.g., '$2100', '$4300') or name (e.g., 'INIDISP')"
                    }
                },
                "required": ["register"]
            },
            handler=tools.lookup_snes_register
        ),
        "lookup_alttp_ram": ToolDefinition(
            name="lookup_alttp_ram",
            description="Look up documentation for an ALTTP/Oracle RAM address",
            parameters={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "RAM address (e.g., '$7E0022', '$7EF3D6') or variable name"
                    }
                },
                "required": ["address"]
            },
            handler=tools.lookup_alttp_ram
        ),
        "lookup_oracle_docs": ToolDefinition(
            name="lookup_oracle_docs",
            description="Search Oracle of Secrets documentation for a topic",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to search for (e.g., 'mask system', 'time system', 'sprites')"
                    }
                },
                "required": ["topic"]
            },
            handler=tools.lookup_oracle_docs
        ),
        "search_oracle_code": ToolDefinition(
            name="search_oracle_code",
            description="Search Oracle of Secrets codebase for a pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File extension to search (default: asm)",
                        "default": "asm"
                    }
                },
                "required": ["pattern"]
            },
            handler=tools.search_oracle_code
        ),
        "check_memory_free": ToolDefinition(
            name="check_memory_free",
            description="Check if a ROM bank/region is available for use",
            parameters={
                "type": "object",
                "properties": {
                    "bank": {
                        "type": "string",
                        "description": "ROM bank (e.g., '2B', '3C')"
                    },
                    "start": {
                        "type": "string",
                        "description": "Start address within bank"
                    },
                    "end": {
                        "type": "string",
                        "description": "End address within bank"
                    }
                },
                "required": ["bank"]
            },
            handler=tools.check_memory_free
        ),
        "get_sprite_table_format": ToolDefinition(
            name="get_sprite_table_format",
            description="Get the Oracle sprite table entry format documentation",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=tools.get_sprite_table_format
        ),
        "validate_namespace": ToolDefinition(
            name="validate_namespace",
            description="Check assembly code for namespace and asar issues",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Assembly code to validate"
                    }
                },
                "required": ["code"]
            },
            handler=tools.validate_namespace
        ),
    }


# Global tools instance
_tools_instance = OracleTools()
ORACLE_TOOLS = _create_tool_defs(_tools_instance)


def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool

    Returns:
        Tool execution result
    """
    if tool_name not in ORACLE_TOOLS:
        return f"Unknown tool: {tool_name}"

    tool = ORACLE_TOOLS[tool_name]
    try:
        return tool.handler(**kwargs)
    except Exception as e:
        return f"Tool error: {e}"
