"""AST-aware codebase symbol indexing for agent navigation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codebase_explorer import (
    _IGNORED_DIRS,
    _LANGUAGE_BY_EXTENSION,
    infer_project_root,
)

_SCHEMA_VERSION = 1
_DEFAULT_MAX_INDEX_FILES = 5000
_DEFAULT_MAX_FILE_BYTES = 1_000_000  # skip files > 1 MB

# Languages with AST or regex extraction support
_SUPPORTED_LANGUAGES = {"python", "typescript", "javascript", "rust", "go"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbolMatch:
    symbol_name: str
    kind: str  # functions | classes | imports | exports
    source_path: str
    language: str
    line_start: int
    line_end: int
    score: float = 1.0
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_name": self.symbol_name,
            "kind": self.kind,
            "source_path": self.source_path,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": self.score,
            "docstring": self.docstring,
        }


@dataclass
class CodebaseIndexResult:
    total_files: int = 0
    indexed: int = 0
    skipped: int = 0
    reused: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
    mode: str = "full"

    def summary(self) -> str:
        parts = [
            f"total={self.total_files} indexed={self.indexed} "
            f"skipped={self.skipped} errors={len(self.errors)}"
        ]
        if self.mode == "incremental":
            parts.append(f"reused={self.reused} removed={self.removed}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_codebase_index(
    project_root: Path,
    output_dir: Path,
    *,
    max_files: int = _DEFAULT_MAX_INDEX_FILES,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    incremental: bool = True,
    languages: list[str] | None = None,
) -> CodebaseIndexResult:
    """Walk *project_root* and build/update a per-file symbol index under *output_dir*.

    Artifacts are written as ``<sha1_of_abs_path>.json`` under *output_dir*.
    A manifest at ``output_dir/index.json`` tracks mtime/size for incremental
    updates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "index.json"
    result = CodebaseIndexResult(mode="incremental" if incremental else "full")

    allowed_langs: set[str] | None = (
        {lang.lower().strip() for lang in languages if lang.strip()} if languages else None
    )

    # Load existing manifest for incremental mode
    old_manifest: dict[str, dict[str, Any]] = {}
    if incremental and manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            old_manifest = raw.get("files", {})
        except (json.JSONDecodeError, OSError):
            old_manifest = {}

    new_manifest: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()

    for abs_path, language in _iter_indexable_files(project_root, max_files, allowed_langs):
        abs_str = str(abs_path)
        seen_paths.add(abs_str)
        result.total_files += 1

        try:
            stat = abs_path.stat()
        except OSError:
            result.skipped += 1
            continue

        if stat.st_size > max_file_bytes:
            result.skipped += 1
            continue

        artifact_name = _artifact_name(abs_str)
        mtime = stat.st_mtime
        size = stat.st_size
        content_hash = ""

        # Incremental: skip unchanged files
        if incremental and abs_str in old_manifest:
            old_entry = old_manifest[abs_str]
            if (
                old_entry.get("mtime") == mtime
                and old_entry.get("size_bytes") == size
                and (output_dir / old_entry.get("artifact", "")).exists()
            ):
                new_manifest[abs_str] = old_entry
                result.reused += 1
                continue

        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"{abs_str}: read error ({exc})")
            result.skipped += 1
            continue

        content_hash = hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()

        try:
            symbols = _extract_symbols(source, language)
        except Exception as exc:
            result.errors.append(f"{abs_str}: extraction error ({exc})")
            symbols = {"functions": [], "classes": [], "imports": [], "exports": []}

        artifact: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "source_path": abs_str,
            "language": language,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "mtime": mtime,
            "size_bytes": size,
            "content_hash": content_hash,
            "symbols": symbols,
        }

        artifact_path = output_dir / artifact_name
        try:
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=True) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            result.errors.append(f"{abs_str}: write error ({exc})")
            result.skipped += 1
            continue

        new_manifest[abs_str] = {
            "artifact": artifact_name,
            "mtime": mtime,
            "size_bytes": size,
            "content_hash": content_hash,
            "language": language,
        }
        result.indexed += 1

    # Remove stale artifacts
    for old_path, old_entry in old_manifest.items():
        if old_path not in seen_paths:
            result.removed += 1
            stale = output_dir / old_entry.get("artifact", "")
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

    manifest_doc: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "files": new_manifest,
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest_doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        result.errors.append(f"manifest write error: {exc}")

    return result


def search_codebase_index(
    output_dir: Path,
    query: str,
    *,
    kind: str | None = None,
    language: str | None = None,
    limit: int = 20,
    exact: bool = False,
) -> list[SymbolMatch]:
    """Search symbol artifacts for *query*.

    *kind* filters to ``functions``, ``classes``, ``imports``, or ``exports``.
    *language* filters to a specific language.
    *exact* requires an exact case-insensitive name match.
    """
    manifest_path = output_dir / "index.json"
    if not manifest_path.exists():
        return []

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    files: dict[str, dict[str, Any]] = raw.get("files", {})
    query_lower = query.lower().strip()
    kind_filter = kind.lower().strip() if kind else None
    lang_filter = language.lower().strip() if language else None

    matches: list[SymbolMatch] = []

    for abs_str, entry in files.items():
        file_lang = entry.get("language", "")
        if lang_filter and file_lang != lang_filter:
            continue

        artifact_name = entry.get("artifact", "")
        if not artifact_name:
            continue

        artifact_path = output_dir / artifact_name
        try:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        symbols: dict[str, list[dict]] = data.get("symbols", {})
        kinds_to_search = [kind_filter] if kind_filter else ["functions", "classes", "imports", "exports"]

        for sym_kind in kinds_to_search:
            for sym in symbols.get(sym_kind, []):
                name = sym.get("name", "")
                if not name:
                    continue
                name_lower = name.lower()
                if exact:
                    if name_lower != query_lower:
                        continue
                    score = 1.0
                else:
                    if query_lower not in name_lower:
                        continue
                    # Prefer exact > prefix > substring
                    if name_lower == query_lower:
                        score = 1.0
                    elif name_lower.startswith(query_lower):
                        score = 0.9
                    else:
                        score = 0.7

                matches.append(
                    SymbolMatch(
                        symbol_name=name,
                        kind=sym_kind,
                        source_path=abs_str,
                        language=file_lang,
                        line_start=sym.get("line_start", 0),
                        line_end=sym.get("line_end", 0),
                        score=score,
                        docstring=sym.get("docstring"),
                    )
                )

    matches.sort(key=lambda m: (-m.score, m.symbol_name, m.source_path))
    return matches[:limit]


def find_symbol_definition(
    output_dir: Path,
    name: str,
    *,
    language: str | None = None,
) -> list[SymbolMatch]:
    """Find exact definition(s) of *name* across all indexed files."""
    return search_codebase_index(
        output_dir,
        name,
        kind=None,
        language=language,
        limit=50,
        exact=True,
    )


def codebase_index_dir(context_path: Path) -> Path:
    """Return the conventional codebase index output dir for a context path."""
    from .context_paths import resolve_mount_root
    from .models import MountType

    try:
        scratchpad = resolve_mount_root(context_path, MountType.SCRATCHPAD)
        return scratchpad / "codebase"
    except Exception:
        return context_path / "scratchpad" / "codebase"


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


def _extract_symbols(source: str, language: str) -> dict[str, list[dict[str, Any]]]:
    """Dispatch to language-specific extractor."""
    if language == "python":
        return _extract_python_symbols(source)
    if language in ("typescript", "javascript"):
        return _extract_ts_js_symbols(source)
    if language == "rust":
        return _extract_rust_symbols(source)
    if language == "go":
        return _extract_go_symbols(source)
    return {"functions": [], "classes": [], "imports": [], "exports": []}


def _extract_python_symbols(source: str) -> dict[str, list[dict[str, Any]]]:
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"functions": functions, "classes": classes, "imports": imports, "exports": []}

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level and class-level methods (depth 0 or 1)
            functions.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [
                        ast.unparse(d) if hasattr(ast, "unparse") else ""
                        for d in node.decorator_list
                    ],
                    "docstring": ast.get_docstring(node),
                }
            )
        elif isinstance(node, ast.ClassDef):
            method_names = [
                n.name
                for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node
            ]
            classes.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "bases": [
                        ast.unparse(b) if hasattr(ast, "unparse") else ""
                        for b in node.bases
                    ],
                    "methods": method_names,
                    "docstring": ast.get_docstring(node),
                }
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "name": alias.asname or alias.name,
                        "module": alias.name,
                        "line_start": node.lineno,
                        "line_end": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(
                    {
                        "name": alias.asname or alias.name,
                        "module": module,
                        "line_start": node.lineno,
                        "line_end": node.lineno,
                    }
                )

    _ = lines  # available for future use
    return {"functions": functions, "classes": classes, "imports": imports, "exports": []}


def _extract_ts_js_symbols(source: str) -> dict[str, list[dict[str, Any]]]:
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []

    lines = source.splitlines()

    # export function / export async function / export default function
    _re_fn = re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)\s*[(<]",
        re.MULTILINE,
    )
    # export const/let/var NAME = ... / export const NAME: Type = ...
    _re_const = re.compile(
        r"^export\s+(?:const|let|var)\s+(\w+)\s*(?::|=)",
        re.MULTILINE,
    )
    # class / export class
    _re_class = re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+(\w+)",
        re.MULTILINE,
    )
    # import ... from '...'
    _re_import = re.compile(
        r"^import\s+(?:type\s+)?(?:\*\s+as\s+(\w+)|(\{[^}]+\})|(\w+))(?:\s*,\s*(?:\{[^}]+\}|\w+))?\s+from\s+['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )

    for match in _re_fn.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        name = match.group(1)
        is_export = match.group(0).startswith("export")
        entry = {"name": name, "line_start": line_no, "line_end": line_no}
        if is_export:
            exports.append(entry)
        functions.append(entry)

    for match in _re_const.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        exports.append({"name": match.group(1), "line_start": line_no, "line_end": line_no})

    for match in _re_class.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        name = match.group(1)
        is_export = match.group(0).startswith("export")
        entry = {"name": name, "line_start": line_no, "line_end": line_no}
        if is_export:
            exports.append(entry)
        classes.append(entry)

    for match in _re_import.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        module = match.group(4) or ""
        star_as = match.group(1)
        named = match.group(2)
        default = match.group(3)
        if star_as:
            imports.append({"name": star_as, "module": module, "line_start": line_no, "line_end": line_no})
        if default:
            imports.append({"name": default, "module": module, "line_start": line_no, "line_end": line_no})
        if named:
            for name in re.findall(r"\b(\w+)\b", named):
                if name not in ("type", "as"):
                    imports.append({"name": name, "module": module, "line_start": line_no, "line_end": line_no})

    _ = lines
    return {"functions": functions, "classes": classes, "imports": imports, "exports": exports}


def _extract_rust_symbols(source: str) -> dict[str, list[dict[str, Any]]]:
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []

    _re_pub_fn = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE)
    _re_pub_struct = re.compile(r"^pub(?:\([^)]*\))?\s+struct\s+(\w+)", re.MULTILINE)
    _re_pub_enum = re.compile(r"^pub(?:\([^)]*\))?\s+enum\s+(\w+)", re.MULTILINE)
    _re_pub_trait = re.compile(r"^pub(?:\([^)]*\))?\s+trait\s+(\w+)", re.MULTILINE)
    _re_use = re.compile(r"^use\s+([\w::{}, ]+)\s*;", re.MULTILINE)

    for match in _re_pub_fn.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        is_pub = "pub" in match.group(0)
        entry = {"name": match.group(1), "line_start": line_no, "line_end": line_no}
        functions.append(entry)
        if is_pub:
            exports.append(entry)

    for pattern, kind in ((_re_pub_struct, "struct"), (_re_pub_enum, "enum"), (_re_pub_trait, "trait")):
        for match in pattern.finditer(source):
            line_no = source[: match.start()].count("\n") + 1
            entry = {"name": match.group(1), "kind": kind, "line_start": line_no, "line_end": line_no}
            classes.append(entry)
            exports.append(entry)

    for match in _re_use.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        imports.append({"name": match.group(1).strip(), "module": match.group(1).strip(), "line_start": line_no, "line_end": line_no})

    return {"functions": functions, "classes": classes, "imports": imports, "exports": exports}


def _extract_go_symbols(source: str) -> dict[str, list[dict[str, Any]]]:
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []

    # func Name( or func (recv Type) Name(
    _re_func = re.compile(r"^func(?:\s+\([^)]+\))?\s+(\w+)\s*\(", re.MULTILINE)
    # type Name struct / type Name interface
    _re_type = re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\s*\{", re.MULTILINE)
    # import "pkg" or import alias "pkg"
    _re_import_single = re.compile(r'^\s*(?:(\w+)\s+)?"([^"]+)"', re.MULTILINE)
    _re_import_block = re.compile(r"^import\s+\(([^)]+)\)", re.MULTILINE | re.DOTALL)

    for match in _re_func.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        name = match.group(1)
        functions.append({"name": name, "line_start": line_no, "line_end": line_no})

    for match in _re_type.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        classes.append({"name": match.group(1), "line_start": line_no, "line_end": line_no})

    for block_match in _re_import_block.finditer(source):
        block = block_match.group(1)
        for imp_match in _re_import_single.finditer(block):
            alias = imp_match.group(1) or ""
            pkg = imp_match.group(2) or ""
            line_no = source[: block_match.start() + imp_match.start()].count("\n") + 1
            name = alias or pkg.split("/")[-1]
            imports.append({"name": name, "module": pkg, "line_start": line_no, "line_end": line_no})

    # Exported names are capitalized in Go
    exported_fns = [f for f in functions if f["name"][:1].isupper()]
    exported_types = [c for c in classes if c["name"][:1].isupper()]

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "exports": exported_fns + exported_types,
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _iter_indexable_files(
    project_root: Path,
    max_files: int,
    allowed_langs: set[str] | None,
) -> list[tuple[Path, str]]:
    results: list[tuple[Path, str]] = []
    count = 0

    for base, dirs, files in os.walk(project_root):
        dirs[:] = sorted(
            name for name in dirs
            if not (name in _IGNORED_DIRS or (name.startswith(".") and name != ".github"))
        )
        for filename in sorted(files):
            ext = Path(filename).suffix.lower()
            language = _LANGUAGE_BY_EXTENSION.get(ext)
            if not language or language not in _SUPPORTED_LANGUAGES:
                continue
            if allowed_langs and language not in allowed_langs:
                continue
            results.append((Path(base) / filename, language))
            count += 1
            if count >= max_files:
                return results

    return results


def _artifact_name(abs_str: str) -> str:
    return hashlib.sha1(abs_str.encode("utf-8")).hexdigest() + ".json"
