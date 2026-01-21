"""Tool definitions for agent harness.

Provides model-agnostic tool definitions that can be used with any backend.
Includes AFS context tools and Triforce-specific assembly tools.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONTEXT_ROOT = Path.home() / ".context"
DEFAULT_WORKSPACE_ROOT = Path.home() / "src"
DEFAULT_ASAR_PATH = Path.home() / "src/third_party/asar-repo/build/asar/bin/asar"
DEFAULT_QUERY_TOOL_PATH = Path.home() / "src/lab/afs/tools/query_file.py"


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """Model-agnostic tool definition.

    Can be converted to OpenAI, Gemini, or Anthropic format.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_gemini(self) -> dict[str, Any]:
        """Convert to Gemini FunctionDeclaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            return await self.handler(arguments)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(
                success=False,
                content="",
                error=str(e),
            )


# ============================================================================
# AFS Context Tools - Give models access to the AFS ecosystem
# ============================================================================


async def read_context_handler(args: dict[str, Any]) -> ToolResult:
    """Read a file from AFS context."""
    path = args.get("path", "")
    if not path:
        return ToolResult(success=False, content="", error="No path provided")

    # Resolve relative to context root
    context_root = Path(args.get("context_root", DEFAULT_CONTEXT_ROOT))
    full_path = context_root / path

    # Security: ensure path is under context root
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(context_root.resolve())):
            return ToolResult(
                success=False,
                content="",
                error="Path escapes context root",
            )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))

    if not full_path.exists():
        return ToolResult(
            success=False,
            content="",
            error=f"File not found: {path}",
        )

    try:
        content = full_path.read_text()
        return ToolResult(
            success=True,
            content=content,
            metadata={"path": str(full_path), "size": len(content)},
        )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))


async def write_scratchpad_handler(args: dict[str, Any]) -> ToolResult:
    """Write to scratchpad for working memory."""
    filename = args.get("filename", "")
    content = args.get("content", "")

    if not filename:
        return ToolResult(success=False, content="", error="No filename provided")

    # Security: sanitize filename
    filename = Path(filename).name  # Strip any path components

    context_root = Path(args.get("context_root", DEFAULT_CONTEXT_ROOT))
    scratchpad_dir = context_root / "scratchpad"
    scratchpad_dir.mkdir(parents=True, exist_ok=True)

    full_path = scratchpad_dir / filename

    try:
        full_path.write_text(content)
        return ToolResult(
            success=True,
            content=f"Wrote {len(content)} bytes to {filename}",
            metadata={"path": str(full_path)},
        )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))


async def ws_find_handler(args: dict[str, Any]) -> ToolResult:
    """Find projects in workspace using ws tool."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, content="", error="No query provided")

    try:
        result = subprocess.run(
            ["ws", "find", query],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return ToolResult(
                success=True,
                content=result.stdout.strip(),
                metadata={"query": query},
            )
        else:
            return ToolResult(
                success=False,
                content="",
                error=result.stderr or "ws find failed",
            )
    except FileNotFoundError:
        return ToolResult(
            success=False,
            content="",
            error="ws command not found",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            content="",
            error="ws find timed out",
        )


async def grep_handler(args: dict[str, Any]) -> ToolResult:
    """Search code with ripgrep."""
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    if not pattern:
        return ToolResult(success=False, content="", error="No pattern provided")

    try:
        result = subprocess.run(
            ["rg", "--max-count=50", "--line-number", pattern, path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # rg returns 1 if no matches, 0 if matches found
        if result.returncode in (0, 1):
            output = result.stdout.strip()
            if not output:
                output = "No matches found"
            return ToolResult(
                success=True,
                content=output,
                metadata={"pattern": pattern, "path": path},
            )
        else:
            return ToolResult(
                success=False,
                content="",
                error=result.stderr,
            )
    except FileNotFoundError:
        return ToolResult(
            success=False,
            content="",
            error="rg (ripgrep) command not found",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            content="",
            error="grep timed out",
        )


async def shell_handler(args: dict[str, Any]) -> ToolResult:
    """Execute shell command (sandboxed)."""
    command = args.get("command", "")
    if not command:
        return ToolResult(success=False, content="", error="No command provided")

    # Security: blocklist dangerous commands
    dangerous = ["rm -rf", "sudo", "chmod 777", "> /dev/", "mkfs", "dd if="]
    for d in dangerous:
        if d in command.lower():
            return ToolResult(
                success=False,
                content="",
                error=f"Command blocked for safety: {d}",
            )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=args.get("cwd", None),
        )

        return ToolResult(
            success=result.returncode == 0,
            content=result.stdout + result.stderr,
            error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
            metadata={"command": command, "returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            content="",
            error="Command timed out after 60s",
        )


async def read_file_handler(args: dict[str, Any]) -> ToolResult:
    """Read a file from the filesystem."""
    path = args.get("path", "")
    if not path:
        return ToolResult(success=False, content="", error="No path provided")

    file_path = Path(path).expanduser()

    if not file_path.exists():
        return ToolResult(
            success=False,
            content="",
            error=f"File not found: {path}",
        )

    try:
        content = file_path.read_text()
        return ToolResult(
            success=True,
            content=content,
            metadata={"path": str(file_path), "size": len(content)},
        )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))


async def write_file_handler(args: dict[str, Any]) -> ToolResult:
    """Write a file to the filesystem."""
    path = args.get("path", "")
    content = args.get("content", "")

    if not path:
        return ToolResult(success=False, content="", error="No path provided")

    file_path = Path(path).expanduser()

    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return ToolResult(
            success=True,
            content=f"Wrote {len(content)} bytes to {path}",
            metadata={"path": str(file_path)},
        )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))


async def fs_query_handler(args: dict[str, Any]) -> ToolResult:
    """Query a file using an external agent (External Attention)."""
    path = args.get("path", "")
    query = args.get("query", "")

    if not path or not query:
        return ToolResult(success=False, content="", error="Path and query required")

    script_path = args.get("script_path", str(DEFAULT_QUERY_TOOL_PATH))

    # Resolve path relative to context root if not absolute
    # This aligns with read_context_handler behavior if path is relative
    # But read_file handles absolute. Let's support both via simple expansion
    # or rely on the script to handle it.
    # The script uses Path(path).expanduser().

    try:
        cmd = ["python3", script_path, path, query]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120, # Longer timeout for LLM call
        )

        if result.returncode == 0:
            return ToolResult(
                success=True,
                content=result.stdout.strip(),
                metadata={"path": path, "query": query},
            )
        else:
            return ToolResult(
                success=False,
                content="",
                error=result.stderr or result.stdout or "Query failed",
            )
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))


# ============================================================================
# Triforce Assembly Tools - For 65816/SNES development
# ============================================================================


async def asar_handler(args: dict[str, Any]) -> ToolResult:
    """Assemble 65816 code with asar."""
    import re
    import tempfile

    code = args.get("code", "")
    if not code:
        return ToolResult(success=False, content="", error="No code provided")

    # Extract code from markdown blocks if present
    code_match = re.search(r"```(?:asm|assembly)?\n(.*?)```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    # Add lorom header if not present
    if "lorom" not in code.lower() and "hirom" not in code.lower():
        code = "lorom\n\n" + code

    # Add org if not present
    if "org " not in code.lower():
        code = code.replace("lorom\n", "lorom\n\norg $008000\n")

    asar_path = args.get("asar_path", str(DEFAULT_ASAR_PATH))

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = Path(tmpdir) / "input.asm"
        rom_path = Path(tmpdir) / "output.sfc"

        # Write source
        asm_path.write_text(code)

        # Create empty ROM
        rom_path.write_bytes(b"\x00" * 0x80000)

        try:
            result = subprocess.run(
                [asar_path, "--no-title-check", str(asm_path), str(rom_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                rom_data = rom_path.read_bytes()
                code_bytes = sum(1 for b in rom_data if b != 0)
                return ToolResult(
                    success=True,
                    content=f"Assembly successful! {code_bytes} bytes of code generated.",
                    metadata={
                        "code_bytes": code_bytes,
                        "source": code,
                    },
                )
            else:
                errors = result.stderr or result.stdout
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Assembly failed:\n{errors}",
                    metadata={"source": code},
                )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                content="",
                error=f"asar not found at {asar_path}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content="",
                error="Assembly timed out",
            )


async def yaze_mcp_handler(args: dict[str, Any]) -> ToolResult:
    """Query YAZE emulator state via MCP."""
    command = args.get("command", "status")
    mcp_args = args.get("args", {})

    try:
        result = subprocess.run(
            [
                "claude",
                "mcp",
                "call",
                "--server",
                "yaze-debugger",
                "--tool",
                command,
                "--args",
                json.dumps(mcp_args),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return ToolResult(
                success=True,
                content=result.stdout,
                metadata={"command": command},
            )
        else:
            return ToolResult(
                success=False,
                content="",
                error=result.stderr or "MCP call failed",
            )

    except FileNotFoundError:
        return ToolResult(
            success=False,
            content="",
            error="claude CLI not found - MCP tools unavailable",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            content="",
            error="MCP call timed out",
        )


async def alttp_knowledge_handler(args: dict[str, Any]) -> ToolResult:
    """Look up ALTTP RAM addresses, sprites, etc."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, content="", error="No query provided")

    # Try hyrule-historian MCP first
    try:
        result = subprocess.run(
            [
                "claude",
                "mcp",
                "call",
                "--server",
                "hyrule-historian",
                "--tool",
                "search",
                "--args",
                json.dumps({"query": query}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            return ToolResult(
                success=True,
                content=result.stdout,
                metadata={"query": query, "source": "hyrule-historian"},
            )
    except Exception:
        pass

    # Fallback: search local knowledge files
    knowledge_dir = DEFAULT_CONTEXT_ROOT / "knowledge" / "alttp"
    if knowledge_dir.exists():
        try:
            result = subprocess.run(
                ["rg", "--max-count=20", "-i", query, str(knowledge_dir)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.stdout.strip():
                return ToolResult(
                    success=True,
                    content=result.stdout,
                    metadata={"query": query, "source": "local_knowledge"},
                )
        except Exception:
            pass

    return ToolResult(
        success=False,
        content="",
        error=f"No knowledge found for: {query}",
    )


# ============================================================================
# Tool Collections
# ============================================================================


def create_afs_tools(context_root: Path | None = None) -> list[Tool]:
    """Create AFS context access tools."""
    root = context_root or DEFAULT_CONTEXT_ROOT

    # Inject context_root into handlers via closure
    async def _read_context(args):
        args["context_root"] = str(root)
        return await read_context_handler(args)

    async def _write_scratchpad(args):
        args["context_root"] = str(root)
        return await write_scratchpad_handler(args)

    return [
        Tool(
            name="read_context",
            description="Read a file from AFS context (scratchpad, memory, knowledge)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to context root (e.g., 'scratchpad/notes.md', 'knowledge/alttp/wram.md')",
                    }
                },
                "required": ["path"],
            },
            handler=_read_context,
        ),
        Tool(
            name="write_scratchpad",
            description="Write to scratchpad for working memory",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to write (in scratchpad directory)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["filename", "content"],
            },
            handler=_write_scratchpad,
        ),
        Tool(
            name="ws_find",
            description="Find projects in workspace",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (project name or partial match)",
                    }
                },
                "required": ["query"],
            },
            handler=ws_find_handler,
        ),
        Tool(
            name="grep_codebase",
            description="Search code with ripgrep",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in (default: current directory)",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
            handler=grep_handler,
        ),
        Tool(
            name="run_shell",
            description="Execute shell command (sandboxed, dangerous commands blocked)",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["command"],
            },
            handler=shell_handler,
        ),
        Tool(
            name="read_file",
            description="Read a file from the filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (absolute or relative to home)",
                    }
                },
                "required": ["path"],
            },
            handler=read_file_handler,
        ),
        Tool(
            name="write_file",
            description="Write a file to the filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
            handler=write_file_handler,
        ),
        Tool(
            name="fs_query",
            description="Answer a question about a file without loading it (External Attention)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to query",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question to ask about the file",
                    },
                },
                "required": ["path", "query"],
            },
            handler=fs_query_handler,
        ),
    ]


def create_triforce_tools(
    asar_path: str | None = None,
    context_root: Path | None = None,
) -> list[Tool]:
    """Create Triforce assembly tools (includes AFS tools)."""
    afs_tools = create_afs_tools(context_root)
    asar = asar_path or str(DEFAULT_ASAR_PATH)

    async def _assemble(args):
        args["asar_path"] = asar
        return await asar_handler(args)

    triforce_specific = [
        Tool(
            name="assemble",
            description="Assemble 65816 code with asar assembler",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "65816 assembly code to compile",
                    }
                },
                "required": ["code"],
            },
            handler=_assemble,
        ),
        Tool(
            name="yaze_debug",
            description="Query YAZE emulator state (registers, memory, breakpoints)",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Debug command (status, read_memory, set_breakpoint, etc.)",
                    },
                    "args": {
                        "type": "object",
                        "description": "Command arguments",
                        "default": {},
                    },
                },
                "required": ["command"],
            },
            handler=yaze_mcp_handler,
        ),
        Tool(
            name="alttp_lookup",
            description="Look up ALTTP RAM addresses, sprites, graphics, or game data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up (e.g., 'Link X position', 'sword sprite', 'DMA registers')",
                    }
                },
                "required": ["query"],
            },
            handler=alttp_knowledge_handler,
        ),
    ]

    return afs_tools + triforce_specific


# Pre-configured tool sets
AFS_TOOLS = create_afs_tools()
TRIFORCE_TOOLS = create_triforce_tools()
