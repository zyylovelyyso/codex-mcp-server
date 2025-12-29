#!/usr/bin/env python3
"""
file_search_mcp: 本地文件查找/读取 MCP 服务器（stdio）。

设计目标：
- 支持在允许的根目录下递归查找文件、读取片段、简单 grep
- 默认只允许访问 MCP_ALLOWED_ROOTS 指定的目录（未设置则仅当前工作目录）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field


mcp = FastMCP("file_search_mcp")


def _allowed_roots() -> list[Path]:
    raw = os.environ.get("MCP_ALLOWED_ROOTS") or os.environ.get("MCP_ALLOWED_ROOT")
    if raw:
        parts = [p for p in raw.split(os.pathsep) if p.strip()]
        roots = [Path(p).expanduser().resolve() for p in parts]
    else:
        roots = [Path.cwd().resolve()]
    return roots


def _resolve_under_roots(user_path: str) -> Path:
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        # default interpret relative to first root
        roots = _allowed_roots()
        p = (roots[0] / p).resolve()
    else:
        p = p.resolve()

    for root in _allowed_roots():
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    raise ValueError(f"路径不允许：{user_path} 不在 MCP_ALLOWED_ROOTS={_allowed_roots()} 下")


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


@dataclass(frozen=True)
class _FindRule:
    include_glob: str | None
    include_regex: re.Pattern[str] | None
    name_substring: str | None
    exclude_glob: str | None
    exclude_regex: re.Pattern[str] | None


def _compile_rule(
    include_glob: str | None,
    include_regex: str | None,
    name_substring: str | None,
    exclude_glob: str | None,
    exclude_regex: str | None,
    case_sensitive: bool,
) -> _FindRule:
    flags = 0 if case_sensitive else re.IGNORECASE
    return _FindRule(
        include_glob=include_glob,
        include_regex=re.compile(include_regex, flags=flags) if include_regex else None,
        name_substring=name_substring if (name_substring and case_sensitive) else (name_substring.lower() if name_substring else None),
        exclude_glob=exclude_glob,
        exclude_regex=re.compile(exclude_regex, flags=flags) if exclude_regex else None,
    )


def _match_name(name: str, rule: _FindRule, case_sensitive: bool) -> bool:
    n = name if case_sensitive else name.lower()
    if rule.exclude_glob and Path(name).match(rule.exclude_glob):
        return False
    if rule.exclude_regex and rule.exclude_regex.search(name):
        return False
    if rule.include_glob and not Path(name).match(rule.include_glob):
        return False
    if rule.include_regex and not rule.include_regex.search(name):
        return False
    if rule.name_substring and rule.name_substring not in n:
        return False
    return True


class FsFindInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    root: str | None = Field(
        default=None,
        description="限定在该根目录下查找（需在 MCP_ALLOWED_ROOTS 内）；为空则遍历 MCP_ALLOWED_ROOTS",
        max_length=500,
    )
    include_glob: str | None = Field(default=None, description="仅匹配文件名 glob（如 *.pdf, **/*.md）", max_length=200)
    include_regex: str | None = Field(default=None, description="仅匹配文件名 regex", max_length=300)
    name_contains: str | None = Field(default=None, description="文件名包含子串", max_length=200)
    exclude_glob: str | None = Field(default=None, description="排除文件名 glob", max_length=200)
    exclude_regex: str | None = Field(default=None, description="排除文件名 regex", max_length=300)
    case_sensitive: bool = Field(default=False, description="regex/contains 是否大小写敏感")
    max_results: int = Field(default=200, ge=1, le=5000, description="最多返回多少条路径")
    max_file_size_bytes: int = Field(default=20_000_000, ge=0, le=2_000_000_000, description="超过该大小的文件不返回")


@mcp.tool(
    name="fs_find",
    annotations={"title": "查找文件路径", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)
def fs_find(params: FsFindInput) -> str:
    rule = _compile_rule(
        include_glob=params.include_glob,
        include_regex=params.include_regex,
        name_substring=params.name_contains,
        exclude_glob=params.exclude_glob,
        exclude_regex=params.exclude_regex,
        case_sensitive=params.case_sensitive,
    )

    roots = [_resolve_under_roots(params.root)] if params.root else _allowed_roots()
    results: list[dict[str, Any]] = []
    for root in roots:
        for p in _iter_files(root):
            try:
                if params.max_file_size_bytes and p.stat().st_size > params.max_file_size_bytes:
                    continue
            except OSError:
                continue
            if not _match_name(p.name, rule, case_sensitive=params.case_sensitive):
                continue
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = str(p)
            results.append({"path": str(p), "relative_to_root": rel, "size_bytes": p.stat().st_size})
            if len(results) >= params.max_results:
                break
        if len(results) >= params.max_results:
            break

    return json.dumps(
        {"roots": [str(r) for r in roots], "count": len(results), "results": results},
        ensure_ascii=False,
        indent=2,
    )


class FsReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    path: str = Field(..., description="要读取的文件路径（需在 MCP_ALLOWED_ROOTS 下）", min_length=1, max_length=800)
    max_bytes: int = Field(default=200_000, ge=1_000, le=20_000_000, description="最多读取多少字节（防止大文件）")
    encoding: str | None = Field(default=None, description="可选：指定文本编码（默认 utf-8）", max_length=50)


@mcp.tool(
    name="fs_read",
    annotations={"title": "读取文件内容片段", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)
def fs_read(params: FsReadInput) -> str:
    p = _resolve_under_roots(params.path)
    data = p.read_bytes()
    truncated = False
    if len(data) > params.max_bytes:
        data = data[: params.max_bytes]
        truncated = True

    enc = params.encoding or "utf-8"
    text = data.decode(enc, errors="replace")
    return json.dumps(
        {"path": str(p), "byte_length": len(data), "truncated": truncated, "encoding": enc, "text": text},
        ensure_ascii=False,
        indent=2,
    )


class FsGrepInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    root: str | None = Field(default=None, description="限定根目录（需在 MCP_ALLOWED_ROOTS 下）", max_length=500)
    include_glob: str | None = Field(default="*", description="仅在匹配该 glob 的文件中搜索", max_length=200)
    pattern: str = Field(..., description="regex 模式", min_length=1, max_length=300)
    case_sensitive: bool = Field(default=False, description="是否大小写敏感")
    max_files: int = Field(default=200, ge=1, le=5000, description="最多搜索多少文件")
    max_hits: int = Field(default=200, ge=1, le=5000, description="最多返回多少条命中")
    max_bytes_per_file: int = Field(default=500_000, ge=1_000, le=20_000_000, description="每个文件最多读取字节数")


@mcp.tool(
    name="fs_grep",
    annotations={"title": "在文件中搜索文本（简易 grep）", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)
def fs_grep(params: FsGrepInput) -> str:
    flags = 0 if params.case_sensitive else re.IGNORECASE
    rx = re.compile(params.pattern, flags=flags)

    roots = [_resolve_under_roots(params.root)] if params.root else _allowed_roots()
    hits: list[dict[str, Any]] = []
    scanned_files = 0

    for root in roots:
        for p in _iter_files(root):
            if params.include_glob and not p.match(params.include_glob):
                continue
            scanned_files += 1
            if scanned_files > params.max_files:
                break
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if len(data) > params.max_bytes_per_file:
                data = data[: params.max_bytes_per_file]
            text = data.decode("utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"path": str(p), "line": i, "text": line[:500]})
                    if len(hits) >= params.max_hits:
                        break
            if len(hits) >= params.max_hits:
                break
        if scanned_files > params.max_files or len(hits) >= params.max_hits:
            break

    return json.dumps(
        {
            "roots": [str(r) for r in roots],
            "scanned_files": scanned_files,
            "hit_count": len(hits),
            "hits": hits,
        },
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()

