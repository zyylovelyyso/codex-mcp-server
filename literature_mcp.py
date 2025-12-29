#!/usr/bin/env python3
"""
literature_mcp: 文献检索与抓取 MCP 服务器（stdio）。

工具覆盖：
- arXiv：关键词检索（返回标题/作者/摘要/arXiv id/pdf 链接）
- Crossref：关键词检索（返回 DOI 等元信息）
- DOI → BibTeX
- 导入引用：BibTeX/RIS（适配 CNKI / Google Scholar 的“导出引用”工作流）
- 下载 PDF/文件到本地（路径受 MCP_ALLOWED_ROOT 限制）

说明：
- 需要联网；在 Codex/客户端启用 MCP 的 open-world 请求后，通常会触发网络访问确认。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from lxml import etree
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator


mcp = FastMCP("literature_mcp")


def _allowed_root() -> Path:
    root = os.environ.get("MCP_ALLOWED_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_under_root(user_path: str) -> Path:
    root = _allowed_root()
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise ValueError(f"路径不允许：{user_path} 不在 MCP_ALLOWED_ROOT={root} 下") from e
    return p


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _read_text_file_under_root(path: str, max_bytes: int = 2_000_000) -> str:
    p = _resolve_under_root(path)
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _parse_bibtex_entries(bibtex: str, max_entries: int) -> list[dict[str, Any]]:
    # Minimal BibTeX parser (no external deps): splits on @...{...}
    # Good enough for CNKI/Scholar exports; not a full BibTeX grammar.
    entries: list[dict[str, Any]] = []
    chunks = re.split(r"(?=@\w+\s*[{(])", bibtex, flags=re.IGNORECASE)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("@"):
            continue
        m = re.match(r"@(?P<type>\w+)\s*[{(]\s*(?P<key>[^,]+)\s*,", chunk, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        entry_type = m.group("type").lower()
        cite_key = m.group("key").strip()
        body = chunk[m.end() :].strip()
        body = re.sub(r"[)}]\s*$", "", body, flags=re.DOTALL).strip()

        fields: dict[str, str] = {}
        # Extract field = {value} / "value" / bareword; handle nested braces shallowly
        i = 0
        while i < len(body):
            m2 = re.search(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_:-]*)\s*=\s*", body[i:])
            if not m2:
                break
            name = m2.group("name").lower()
            j = i + m2.end()
            if j >= len(body):
                break

            value = ""
            if body[j] in ['"', "{"]:
                quote_char = body[j]
                j += 1
                depth = 1 if quote_char == "{" else 0
                while j < len(body):
                    ch = body[j]
                    if quote_char == "{":
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                j += 1
                                break
                        value += ch
                        j += 1
                    else:
                        if ch == '"':
                            j += 1
                            break
                        value += ch
                        j += 1
                value = value.strip()
            else:
                m3 = re.match(r"([^,]+)", body[j:])
                value = (m3.group(1) if m3 else "").strip()
                j += len(value)

            fields[name] = value.strip().strip(",")
            comma = body.find(",", j)
            if comma == -1:
                break
            i = comma + 1

        doi = fields.get("doi") or fields.get("DOI")
        entries.append({"entry_type": entry_type, "cite_key": cite_key, "fields": fields, "doi": doi})
        if len(entries) >= max_entries:
            break
    return entries


def _parse_ris_entries(ris: str, max_entries: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in ris.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("ER  -"):
            if current:
                doi = None
                for k in ("DO", "doi", "DOI"):
                    if k in current:
                        doi = current[k][0] if isinstance(current[k], list) else current[k]
                        break
                entries.append({"fields": current, "doi": doi})
                current = {}
                if len(entries) >= max_entries:
                    break
            continue
        m = re.match(r"^(?P<tag>[A-Z0-9]{2})\s*-\s*(?P<val>.*)$", line)
        if not m:
            continue
        tag = m.group("tag")
        val = m.group("val").strip()
        if tag in current:
            if isinstance(current[tag], list):
                current[tag].append(val)
            else:
                current[tag] = [current[tag], val]
        else:
            current[tag] = val
    return entries


class ArxivSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    query: str = Field(..., description="arXiv 搜索关键词（建议英文）", min_length=1, max_length=300)
    max_results: int = Field(default=10, ge=1, le=50, description="返回条数")
    start: int = Field(default=0, ge=0, le=10_000, description="起始偏移（分页）")
    sort_by: str = Field(default="relevance", description="relevance/lastUpdatedDate/submittedDate", max_length=50)
    timeout_seconds: float = Field(default=30, ge=1, le=120, description="超时秒数")

    @field_validator("sort_by")
    @classmethod
    def _validate_sort_by(cls, v: str) -> str:
        if v not in {"relevance", "lastUpdatedDate", "submittedDate"}:
            raise ValueError("sort_by 仅支持 relevance/lastUpdatedDate/submittedDate")
        return v


@mcp.tool(
    name="lit_search_arxiv",
    annotations={
        "title": "arXiv 文献检索",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def lit_search_arxiv(params: ArxivSearchInput) -> str:
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{quote(params.query)}"
        f"&start={params.start}&max_results={params.max_results}&sortBy={params.sort_by}"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(params.timeout_seconds)) as client:
        resp = await client.get(url, headers={"User-Agent": "literature_mcp/1.0"})
        resp.raise_for_status()

    doc = etree.fromstring(resp.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in doc.findall("a:entry", namespaces=ns):
        title = (entry.findtext("a:title", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", namespaces=ns) or "").strip()
        updated = (entry.findtext("a:updated", namespaces=ns) or "").strip()
        id_url = (entry.findtext("a:id", namespaces=ns) or "").strip()
        authors = [a.findtext("a:name", namespaces=ns) for a in entry.findall("a:author", namespaces=ns)]
        authors = [a for a in authors if a]
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else None
        pdf_url = None
        for link in entry.findall("a:link", namespaces=ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        entries.append(
            {
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "updated": updated,
                "id_url": id_url,
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
            }
        )

    return json.dumps({"query": params.query, "count": len(entries), "results": entries}, ensure_ascii=False, indent=2)


class CrossrefSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    query: str = Field(..., description="Crossref 搜索关键词", min_length=1, max_length=300)
    rows: int = Field(default=10, ge=1, le=50, description="返回条数")
    offset: int = Field(default=0, ge=0, le=10_000, description="起始偏移（分页）")
    timeout_seconds: float = Field(default=30, ge=1, le=120, description="超时秒数")


@mcp.tool(
    name="lit_search_crossref",
    annotations={
        "title": "Crossref 文献检索（DOI/元信息）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def lit_search_crossref(params: CrossrefSearchInput) -> str:
    url = (
        "https://api.crossref.org/works?"
        f"query={quote(params.query)}&rows={params.rows}&offset={params.offset}"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(params.timeout_seconds)) as client:
        resp = await client.get(url, headers={"User-Agent": "literature_mcp/1.0 (mailto:example@example.com)"})
        resp.raise_for_status()
        data = resp.json()

    items = data.get("message", {}).get("items", [])
    results = []
    for it in items:
        title = (it.get("title") or [None])[0]
        container = (it.get("container-title") or [None])[0]
        doi = it.get("DOI")
        issued = None
        issued_parts = ((it.get("issued") or {}).get("date-parts") or [])
        if issued_parts and issued_parts[0]:
            issued = "-".join(str(x) for x in issued_parts[0])
        author = []
        for a in it.get("author") or []:
            family = a.get("family") or ""
            given = a.get("given") or ""
            name = (given + " " + family).strip()
            if name:
                author.append(name)
        results.append(
            {"title": title, "container_title": container, "doi": doi, "issued": issued, "authors": author[:20]}
        )

    return json.dumps({"query": params.query, "count": len(results), "results": results}, ensure_ascii=False, indent=2)


class DoiBibtexInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    doi: str = Field(..., description="DOI（例如 10.1038/nature12373）", min_length=3, max_length=200)
    timeout_seconds: float = Field(default=30, ge=1, le=120, description="超时秒数")


@mcp.tool(
    name="lit_get_bibtex",
    annotations={
        "title": "DOI 转 BibTeX",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def lit_get_bibtex(params: DoiBibtexInput) -> str:
    doi = params.doi.strip()
    url = f"https://doi.org/{doi}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(params.timeout_seconds), follow_redirects=True) as client:
        resp = await client.get(url, headers={"Accept": "application/x-bibtex"})
        resp.raise_for_status()
        bib = resp.text.strip()
    return json.dumps({"doi": doi, "bibtex": bib}, ensure_ascii=False, indent=2)


class DownloadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    url: str = Field(..., description="要下载的链接（http/https）", min_length=1, max_length=2000)
    save_to: str = Field(..., description="保存到该路径（相对路径优先，需在 MCP_ALLOWED_ROOT 下）", min_length=1, max_length=800)
    timeout_seconds: float = Field(default=60, ge=1, le=300, description="超时秒数")
    max_bytes: int = Field(default=20_000_000, ge=10_000, le=200_000_000, description="最大下载字节数")


@mcp.tool(
    name="lit_download",
    annotations={
        "title": "下载 PDF/文件到本地",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def lit_download(params: DownloadInput) -> str:
    p = _resolve_under_root(params.save_to)
    p.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(params.timeout_seconds), follow_redirects=True) as client:
        resp = await client.get(params.url, headers={"User-Agent": "literature_mcp/1.0"})
        resp.raise_for_status()
        content = resp.content[: params.max_bytes]
        truncated = len(resp.content) > len(content)

    p.write_bytes(content)
    return json.dumps(
        {
            "url": params.url,
            "final_url": str(resp.url),
            "saved_to": str(p),
            "content_type": resp.headers.get("content-type", ""),
            "byte_length": len(content),
            "truncated_by_max_bytes": truncated,
            "sha256": _sha256(content) if content else None,
        },
        ensure_ascii=False,
        indent=2,
    )


class ImportBibtexTextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=False, validate_assignment=True, extra="forbid")

    bibtex: str = Field(..., description="BibTeX 原文（可直接粘贴 Scholar/CNKI 导出的 BibTeX）", min_length=1)
    max_entries: int = Field(default=100, ge=1, le=2000, description="最多解析多少条")


@mcp.tool(
    name="lit_import_bibtex_text",
    annotations={
        "title": "导入 BibTeX（文本）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def lit_import_bibtex_text(params: ImportBibtexTextInput) -> str:
    entries = _parse_bibtex_entries(params.bibtex, max_entries=params.max_entries)
    return json.dumps({"count": len(entries), "results": entries}, ensure_ascii=False, indent=2)


class ImportBibtexFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    path: str = Field(..., description="BibTeX 文件路径（需在 MCP_ALLOWED_ROOT 下）", min_length=1, max_length=800)
    max_entries: int = Field(default=500, ge=1, le=5000, description="最多解析多少条")
    max_bytes: int = Field(default=2_000_000, ge=10_000, le=50_000_000, description="最多读取多少字节")


@mcp.tool(
    name="lit_import_bibtex_file",
    annotations={
        "title": "导入 BibTeX（文件）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def lit_import_bibtex_file(params: ImportBibtexFileInput) -> str:
    text = _read_text_file_under_root(params.path, max_bytes=params.max_bytes)
    entries = _parse_bibtex_entries(text, max_entries=params.max_entries)
    return json.dumps({"path": params.path, "count": len(entries), "results": entries}, ensure_ascii=False, indent=2)


class ImportRisFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    path: str = Field(..., description="RIS 文件路径（需在 MCP_ALLOWED_ROOT 下）", min_length=1, max_length=800)
    max_entries: int = Field(default=500, ge=1, le=5000, description="最多解析多少条")
    max_bytes: int = Field(default=5_000_000, ge=10_000, le=50_000_000, description="最多读取多少字节")


@mcp.tool(
    name="lit_import_ris_file",
    annotations={
        "title": "导入 RIS（文件，适配 CNKI/EndNote/RefMan 等导出）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def lit_import_ris_file(params: ImportRisFileInput) -> str:
    text = _read_text_file_under_root(params.path, max_bytes=params.max_bytes)
    entries = _parse_ris_entries(text, max_entries=params.max_entries)
    return json.dumps({"path": params.path, "count": len(entries), "results": entries}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
