#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field


def looks_like_vizkb_root(path: Path) -> bool:
    return bool((path / "tools" / "search.py").exists() and (path / "kb").exists())


def kb_root() -> Path:
    env = os.getenv("VIZKB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[1]
    if looks_like_vizkb_root(candidate):
        return candidate
    # Fallback: search for a sibling folder named "ai-viz-kb" up the tree.
    for p in [candidate, *candidate.parents]:
        maybe = p / "ai-viz-kb"
        if looks_like_vizkb_root(maybe):
            return maybe.resolve()
    return candidate


def vizkb_python() -> str:
    """Python executable used to run VizKB CLI tools.

    By default, uses the same interpreter running this MCP server. Override via:
      - VIZKB_PYTHON=/abs/path/to/python
    """
    env = os.getenv("VIZKB_PYTHON")
    if env:
        return str(Path(env).expanduser())
    return sys.executable


async def _run_python(args: list[str], cwd: Path, timeout_s: float = 60.0) -> tuple[int, str, str]:
    python_exe = vizkb_python()
    proc = await asyncio.create_subprocess_exec(
        python_exe,
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return int(proc.returncode or 0), stdout_b.decode("utf-8", errors="replace"), stderr_b.decode("utf-8", errors="replace")


def _as_error(message: str, *, command: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": message}
    if command is not None:
        out["command"] = command
    return out


def _parse_json(text: str, *, context: str) -> dict[str, Any] | list[Any]:
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from {context}: {e}") from e


mcp = FastMCP("vizkb_mcp")


class Mode(str, Enum):
    recipes = "recipes"
    sources = "sources"
    all = "all"


class Language(str, Enum):
    python = "python"
    r = "r"
    notebook = "notebook"


class Stack(str, Enum):
    matplotlib = "matplotlib"
    seaborn = "seaborn"
    plotnine = "plotnine"
    ggplot2 = "ggplot2"


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_root: str
    warning: str | None = None
    recipe_index: dict[str, Any]
    sources_index: dict[str, Any]
    semantic_index: dict[str, Any]
    hints: dict[str, str]


@mcp.tool(
    name="vizkb_status",
    annotations={
        "title": "VizKB Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_status() -> StatusResponse:
    """Report whether required indexes exist and how to build them."""
    root = kb_root()
    warning = None
    if not looks_like_vizkb_root(root):
        warning = (
            "VIZKB_ROOT is not set (or points to a non-VizKB folder). "
            "Set VIZKB_ROOT to the ai-viz-kb folder path."
        )

    recipe_index = root / "kb" / "index.json"
    sources_index = root / "kb" / "sources_index.json"
    semantic_dir = root / "kb" / "semantic_index"

    def file_status(p: Path) -> dict[str, Any]:
        if not p.exists():
            return {"exists": False}
        st = p.stat()
        return {"exists": True, "size_bytes": int(st.st_size), "mtime": float(st.st_mtime)}

    semantic_config = semantic_dir / "config.json"
    semantic = file_status(semantic_dir / "matrix.npz")
    semantic["dir"] = str(semantic_dir)
    semantic["config_exists"] = semantic_config.exists()
    if semantic_config.exists():
        try:
            semantic["config"] = json.loads(semantic_config.read_text(encoding="utf-8"))
        except Exception:
            semantic["config"] = None

    return StatusResponse(
        kb_root=str(root),
        warning=warning,
        recipe_index=file_status(recipe_index),
        sources_index=file_status(sources_index),
        semantic_index=semantic,
        hints={
            "build_recipe_index": f'python3 "{(root / "tools" / "build_index.py").as_posix()}"',
            "build_sources_index": f'python3 "{(root / "tools" / "build_sources_index.py").as_posix()}"',
            "build_semantic_index": f'python3 "{(root / "tools" / "build_semantic_index.py").as_posix()}" --mode all',
        },
    )


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., description="Free-text query.", min_length=1, max_length=500)
    mode: Mode = Field(default=Mode.recipes, description="Search in recipes, sources, or both.")
    category: str | None = Field(default=None, description="Recipe category slug filter.")
    tags: list[str] = Field(default_factory=list, description="Filter by tags (repeatable).")
    charts: list[str] = Field(default_factory=list, description="Filter by chart types (repeatable).")
    language: Language | None = Field(default=None, description="Implementation language filter.")
    stack: Stack | None = Field(default=None, description="Implementation stack filter.")
    source_root: str | None = Field(default=None, description="Filter sources by root_label.")
    limit: int = Field(default=10, ge=1, le=50)
    use_glossary: bool = Field(default=True, description="Enable glossary-based query expansion.")
    explain: bool = Field(default=False, description="Include query groups used for matching.")


@mcp.tool(
    name="vizkb_search",
    annotations={
        "title": "VizKB Keyword Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_search(params: SearchInput) -> dict[str, Any]:
    """Keyword+glossary search over recipe metadata and indexed sources (structured JSON output)."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    script = root / "tools" / "search.py"
    cmd = [str(script), "--mode", params.mode.value, "--query", params.query, "--limit", str(params.limit), "--json"]
    if not params.use_glossary:
        cmd.append("--no-glossary")
    if params.explain:
        cmd.append("--explain")
    if params.category:
        cmd.extend(["--category", params.category])
    for t in params.tags:
        cmd.extend(["--tag", t])
    for c in params.charts:
        cmd.extend(["--chart", c])
    if params.language:
        cmd.extend(["--language", params.language.value])
    if params.stack:
        cmd.extend(["--stack", params.stack.value])
    if params.source_root:
        cmd.extend(["--source-root", params.source_root])

    code, stdout, stderr = await _run_python(cmd, cwd=root, timeout_s=60.0)
    if code != 0:
        return _as_error((stderr or stdout).strip() or f"search.py exited with code {code}", command=cmd)
    payload = _parse_json(stdout, context="vizkb_search")
    return {"ok": True, "result": payload, "command": cmd}


class SemanticSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., description="Free-text query.", min_length=1, max_length=500)
    mode: Mode = Field(default=Mode.all, description="Search in recipes, sources, or both.")
    limit: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    use_glossary: bool = Field(default=True, description="Enable glossary-based query expansion.")
    explain: bool = Field(default=False, description="Include query expansion groups in output.")


@mcp.tool(
    name="vizkb_semantic_search",
    annotations={
        "title": "VizKB Semantic Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_semantic_search(params: SemanticSearchInput) -> dict[str, Any]:
    """Semantic (TF-IDF hybrid) search over recipes/sources (requires built semantic index)."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    script = root / "tools" / "semantic_search.py"
    cmd = [
        str(script),
        "--query",
        params.query,
        "--mode",
        params.mode.value,
        "--limit",
        str(params.limit),
        "--min-score",
        str(params.min_score),
        "--json",
    ]
    if not params.use_glossary:
        cmd.append("--no-glossary")
    if params.explain:
        cmd.append("--explain")

    code, stdout, stderr = await _run_python(cmd, cwd=root, timeout_s=60.0)
    if code != 0:
        return _as_error((stderr or stdout).strip() or f"semantic_search.py exited with code {code}", command=cmd)
    payload = _parse_json(stdout, context="vizkb_semantic_search")
    return {"ok": True, "result": payload, "command": cmd}


class Stage(str, Enum):
    eda = "eda"
    model = "model"
    evaluate = "evaluate"
    decide = "decide"
    writeup = "writeup"


class DataType(str, Enum):
    tabular = "tabular"
    time_series = "time-series"
    geo = "geo"
    graph = "graph"


class Problem(str, Enum):
    overview = "overview"
    relationship = "relationship"
    regression = "regression"
    classification = "classification"
    forecasting = "forecasting"
    optimization = "optimization"
    multiobjective = "multiobjective"
    sensitivity = "sensitivity"
    scheduling = "scheduling"


class RecommendInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stage: Stage
    data_type: DataType
    problem: Problem
    top: int = Field(default=5, ge=1, le=10)
    include_sources: bool = Field(default=True)
    sources_limit: int = Field(default=2, ge=0, le=10)


@mcp.tool(
    name="vizkb_recommend",
    annotations={
        "title": "VizKB Recommend Recipes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_recommend(params: RecommendInput) -> dict[str, Any]:
    """Recommend recipe IDs given an MCM/ICM scenario (task-first decision rules)."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    script = root / "tools" / "recommend.py"
    cmd = [
        str(script),
        "--stage",
        params.stage.value,
        "--data-type",
        params.data_type.value,
        "--problem",
        params.problem.value,
        "--top",
        str(params.top),
        "--json",
        "--sources-limit",
        str(params.sources_limit),
    ]
    if not params.include_sources:
        cmd.append("--no-sources")

    code, stdout, stderr = await _run_python(cmd, cwd=root, timeout_s=30.0)
    if code != 0:
        return _as_error((stderr or stdout).strip() or f"recommend.py exited with code {code}", command=cmd)
    payload = _parse_json(stdout, context="vizkb_recommend")
    return {"ok": True, "result": payload, "command": cmd}


class ValidateCsvInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recipe_id: str = Field(..., min_length=1, max_length=200)
    csv_path: str = Field(..., description="Path to CSV file.")
    case_insensitive: bool = Field(default=False)


@mcp.tool(
    name="vizkb_validate_csv",
    annotations={
        "title": "VizKB Validate CSV Input",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_validate_csv(params: ValidateCsvInput) -> dict[str, Any]:
    """Validate a CSV file against recipe input requirements (returns structured JSON)."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    script = root / "tools" / "validate_input.py"
    cmd = [str(script), "--recipe-id", params.recipe_id, "--csv", params.csv_path, "--json"]
    if params.case_insensitive:
        cmd.append("--case-insensitive")

    code, stdout, stderr = await _run_python(cmd, cwd=root, timeout_s=30.0)
    if stdout.strip():
        try:
            payload = _parse_json(stdout, context="vizkb_validate_csv")
        except Exception:
            payload = {"raw_stdout": stdout}
    else:
        payload = {"raw_stderr": stderr.strip(), "exit_code": code}

    if code != 0:
        return {"ok": False, "error": (stderr or "").strip() or "Validation failed.", "result": payload, "command": cmd}
    return {"ok": True, "result": payload, "command": cmd}


class BuildSemanticIndexInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Mode = Field(default=Mode.all)
    include_source_code: bool = Field(default=True)
    max_chars_per_doc: int = Field(default=20000, ge=1000, le=200000)
    timeout_s: float = Field(default=240.0, ge=10.0, le=1800.0)


@mcp.tool(
    name="vizkb_build_semantic_index",
    annotations={
        "title": "VizKB Build Semantic Index",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_build_semantic_index(params: BuildSemanticIndexInput) -> dict[str, Any]:
    """Build (or refresh) the semantic index used by vizkb_semantic_search."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    script = root / "tools" / "build_semantic_index.py"
    cmd = [
        str(script),
        "--mode",
        params.mode.value,
        "--max-chars-per-doc",
        str(int(params.max_chars_per_doc)),
    ]
    if not params.include_source_code:
        cmd.append("--no-code")

    code, stdout, stderr = await _run_python(cmd, cwd=root, timeout_s=float(params.timeout_s))
    if code != 0:
        return _as_error((stderr or stdout).strip() or f"build_semantic_index.py exited with code {code}", command=cmd)
    return {"ok": True, "stdout": stdout.strip(), "command": cmd}


class GetRecipeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recipe_id: str = Field(..., min_length=1, max_length=200)
    include_markdown: bool = Field(default=False, description="Include recipe.md and qc.md content.")
    max_chars: int = Field(default=8000, ge=500, le=50000)


@mcp.tool(
    name="vizkb_get_recipe",
    annotations={
        "title": "VizKB Get Recipe Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def vizkb_get_recipe(params: GetRecipeInput) -> dict[str, Any]:
    """Fetch a recipe record from kb/index.json and resolve its local file paths."""
    root = kb_root()
    if not looks_like_vizkb_root(root):
        return _as_error("Invalid VIZKB_ROOT. Set env VIZKB_ROOT to the ai-viz-kb folder path.")
    index_path = root / "kb" / "index.json"
    if not index_path.exists():
        return _as_error(f"Missing recipe index: {index_path}. Run build_index.py first.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    recipes = index.get("recipes") or []
    rec: dict[str, Any] | None = None
    for r in recipes:
        if isinstance(r, dict) and r.get("id") == params.recipe_id:
            rec = r
            break
    if rec is None:
        return _as_error(f"Unknown recipe id: {params.recipe_id}")

    rel = rec.get("path")
    if not isinstance(rel, str) or not rel:
        return {"ok": True, "recipe": rec}

    recipe_dir = (root / rel).resolve()
    py_entry = None
    r_entry = None
    for impl in rec.get("implementations") or []:
        if not isinstance(impl, dict):
            continue
        if impl.get("language") == "python" and isinstance(impl.get("entrypoint"), str):
            py_entry = str((recipe_dir / impl["entrypoint"]).resolve())
        if impl.get("language") == "r" and isinstance(impl.get("entrypoint"), str):
            r_entry = str((recipe_dir / impl["entrypoint"]).resolve())

    out: dict[str, Any] = {
        "ok": True,
        "recipe": rec,
        "paths": {
            "recipe_dir": str(recipe_dir),
            "recipe_md": str((recipe_dir / "recipe.md").resolve()),
            "qc_md": str((recipe_dir / "qc.md").resolve()),
            "python_entrypoint": py_entry,
            "r_entrypoint": r_entry,
        },
    }

    if params.include_markdown:
        def read_if_exists(p: Path) -> str:
            if not p.exists():
                return ""
            return p.read_text(encoding="utf-8", errors="ignore")[: int(params.max_chars)]

        out["markdown"] = {
            "recipe_md": read_if_exists(recipe_dir / "recipe.md"),
            "qc_md": read_if_exists(recipe_dir / "qc.md"),
        }
    return out


def main() -> None:
    # Default transport: stdio (for local clients like Codex CLI).
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
