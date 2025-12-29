#!/usr/bin/env python3
"""
web_excel_mcp: 一个同时提供“网页抓取”和“Excel 读取/预览/导出”的 MCP 服务器（stdio）。

设计目标：
- 工具名清晰、可发现（web_* / excel_* 前缀）
- 返回内容可控（限制输出大小/行数，避免一口气吐超大表）
- 默认只允许访问工作目录内文件（可用 MCP_ALLOWED_ROOT 调整）
"""

import asyncio
import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator


mcp = FastMCP("web_excel_mcp")


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


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


class WebFetchInput(BaseModel):
    """网页抓取参数（支持保存到文件）。"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    url: str = Field(..., description="要抓取的 URL（仅支持 http/https）", min_length=1, max_length=2000)
    method: str = Field(default="GET", description="HTTP 方法：GET/HEAD", pattern=r"^(GET|HEAD)$")
    timeout_seconds: float = Field(default=30, description="超时（秒）", ge=1, le=120)
    headers: Optional[Dict[str, str]] = Field(default=None, description="可选请求头（字典）")
    follow_redirects: bool = Field(default=True, description="是否跟随重定向")
    max_bytes: int = Field(
        default=1_000_000,
        description="最多读取响应体字节数（防止超大内容）",
        ge=10_000,
        le=20_000_000,
    )
    extract_text: bool = Field(
        default=True,
        description="若为 HTML，是否提取可读文本（去标签）。false 时返回原始文本片段。",
    )
    save_to: Optional[str] = Field(
        default=None,
        description="可选：将响应体保存到该路径（相对路径优先，需在 MCP_ALLOWED_ROOT 下）",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="返回格式：json 或 text")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        u = urlparse(v)
        if u.scheme not in {"http", "https"}:
            raise ValueError("url 仅支持 http/https")
        if not u.netloc:
            raise ValueError("url 缺少主机名")
        return v


def _html_to_text(html: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    meta: dict[str, Any] = {"title": title, "truncated": truncated, "text_length": len(text)}
    return text, meta


@mcp.tool(
    name="web_fetch",
    annotations={
        "title": "抓取网页/下载内容",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def web_fetch(params: WebFetchInput) -> str:
    """
    抓取一个 URL（GET/HEAD），可选保存响应体到本地文件，并返回状态/头信息/文本摘要。

    适用：
    - 获取网页 HTML/文本内容
    - 下载 PDF/图片/数据文件到项目目录再进行后续处理
    """

    try:
        async with httpx.AsyncClient(
            follow_redirects=params.follow_redirects,
            timeout=httpx.Timeout(params.timeout_seconds),
            headers=params.headers,
        ) as client:
            resp = await client.request(params.method, params.url)

        content = resp.content[: params.max_bytes]
        truncated = len(resp.content) > len(content)
        content_type = resp.headers.get("content-type", "")

        saved_path: str | None = None
        if params.save_to:
            p = _resolve_under_root(params.save_to)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            saved_path = str(p)

        text_preview: str | None = None
        text_meta: dict[str, Any] | None = None
        if params.method == "GET" and content:
            if "text/html" in content_type and params.extract_text:
                html = content.decode(resp.encoding or "utf-8", errors="replace")
                text_preview, text_meta = _html_to_text(html, max_chars=40_000)
            elif content_type.startswith("text/") or content_type.endswith("json"):
                text_preview = content.decode(resp.encoding or "utf-8", errors="replace")
                if len(text_preview) > 40_000:
                    text_preview = text_preview[:40_000]
                    text_meta = {"truncated": True, "text_length": len(text_preview)}

        result = {
            "requested_url": params.url,
            "final_url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": content_type,
            "headers": {k.lower(): v for k, v in resp.headers.items()},
            "byte_length": len(content),
            "truncated_by_max_bytes": truncated,
            "sha256": _sha256(content) if content else None,
            "saved_to": saved_path,
            "text_preview": text_preview,
            "text_meta": text_meta,
        }

        if params.response_format == ResponseFormat.TEXT:
            lines = [
                f"status={result['status_code']} content_type={content_type}",
                f"final_url={result['final_url']}",
                f"bytes={result['byte_length']} truncated={result['truncated_by_max_bytes']}",
            ]
            if saved_path:
                lines.append(f"saved_to={saved_path}")
            if text_preview:
                lines.append("")
                lines.append(text_preview)
            return "\n".join(lines).strip()

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: web_fetch 失败：{type(e).__name__}: {e}"


class ExcelPathInput(BaseModel):
    """Excel 文件参数（路径受 MCP_ALLOWED_ROOT 限制）。"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    path: str = Field(..., description="Excel 文件路径（.xlsx/.xlsm），相对路径优先", min_length=1, max_length=500)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        p = Path(v)
        if p.suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("仅支持 .xlsx / .xlsm")
        return v


class ExcelListSheetsInput(ExcelPathInput):
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="返回格式：json 或 text")


@mcp.tool(
    name="excel_list_sheets",
    annotations={
        "title": "列出 Excel 工作表",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def excel_list_sheets(params: ExcelListSheetsInput) -> str:
    """列出 Excel 文件的所有工作表名。"""
    try:
        path = _resolve_under_root(params.path)
        sheet_names = await asyncio.to_thread(lambda: pd.ExcelFile(path).sheet_names)
        result = {"path": str(path), "sheet_count": len(sheet_names), "sheets": sheet_names}
        if params.response_format == ResponseFormat.TEXT:
            return "\n".join([f"path={result['path']}", *sheet_names]).strip()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: excel_list_sheets 失败：{type(e).__name__}: {e}"


class ExcelReadSheetInput(ExcelPathInput):
    sheet: str = Field(..., description="工作表名（例如 'Teams'）", min_length=1, max_length=200)
    nrows: int = Field(default=50, description="读取前 n 行（用于预览）", ge=1, le=5000)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="返回格式：json 或 text")


@mcp.tool(
    name="excel_read_sheet_preview",
    annotations={
        "title": "预览 Excel 工作表",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def excel_read_sheet_preview(params: ExcelReadSheetInput) -> str:
    """读取某个工作表的前 n 行，返回列名与预览数据。"""
    try:
        path = _resolve_under_root(params.path)

        def _read() -> pd.DataFrame:
            return pd.read_excel(path, sheet_name=params.sheet, nrows=params.nrows)

        df = await asyncio.to_thread(_read)
        df = df.where(pd.notna(df), None)

        preview = df.to_dict(orient="records")
        result = {"path": str(path), "sheet": params.sheet, "nrows": int(len(df)), "columns": list(df.columns), "rows": preview}

        if params.response_format == ResponseFormat.TEXT:
            table_text = df.to_string(index=False)
            return "\n".join([f"path={result['path']}", f"sheet={params.sheet}", "", table_text]).strip()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: excel_read_sheet_preview 失败：{type(e).__name__}: {e}"


class ExcelProfileInput(ExcelPathInput):
    sheet: str = Field(..., description="工作表名（例如 'Teams'）", min_length=1, max_length=200)
    sample_rows: int = Field(default=5000, description="最多读取前 sample_rows 行做统计", ge=50, le=200_000)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="返回格式：json 或 text")


@mcp.tool(
    name="excel_profile_sheet",
    annotations={
        "title": "统计 Excel 工作表结构",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def excel_profile_sheet(params: ExcelProfileInput) -> str:
    """对工作表做快速画像：列名、类型、缺失数、唯一值数量（基于样本）。"""
    try:
        path = _resolve_under_root(params.path)

        def _read() -> pd.DataFrame:
            return pd.read_excel(path, sheet_name=params.sheet, nrows=params.sample_rows)

        df = await asyncio.to_thread(_read)

        cols = []
        for c in df.columns:
            s = df[c]
            non_null = int(s.notna().sum())
            nulls = int(s.isna().sum())
            uniq = int(s.dropna().astype(str).nunique())
            examples = s.dropna().astype(str).head(3).tolist()
            cols.append(
                {
                    "name": str(c),
                    "dtype": str(s.dtype),
                    "non_null": non_null,
                    "nulls": nulls,
                    "unique_str": uniq,
                    "examples": examples,
                }
            )

        result = {
            "path": str(path),
            "sheet": params.sheet,
            "rows_sampled": int(len(df)),
            "columns": cols,
        }
        if params.response_format == ResponseFormat.TEXT:
            lines = [f"path={result['path']}", f"sheet={params.sheet}", f"rows_sampled={result['rows_sampled']}", ""]
            for c in cols:
                lines.append(
                    f"- {c['name']}: dtype={c['dtype']} non_null={c['non_null']} nulls={c['nulls']} unique≈{c['unique_str']} examples={c['examples']}"
                )
            return "\n".join(lines).strip()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: excel_profile_sheet 失败：{type(e).__name__}: {e}"


class ExcelExportCsvInput(ExcelPathInput):
    sheet: str = Field(..., description="工作表名", min_length=1, max_length=200)
    out_csv: str = Field(..., description="导出的 CSV 路径（相对路径优先）", min_length=1, max_length=500)
    include_index: bool = Field(default=False, description="CSV 是否包含索引列")


@mcp.tool(
    name="excel_export_sheet_csv",
    annotations={
        "title": "导出工作表为 CSV",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def excel_export_sheet_csv(params: ExcelExportCsvInput) -> str:
    """把工作表导出成 CSV（写入 out_csv）。"""
    try:
        in_path = _resolve_under_root(params.path)
        out_path = _resolve_under_root(params.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _export() -> tuple[int, int]:
            df = pd.read_excel(in_path, sheet_name=params.sheet)
            df.to_csv(out_path, index=params.include_index)
            return int(len(df)), int(len(df.columns))

        nrows, ncols = await asyncio.to_thread(_export)
        result = {"in_path": str(in_path), "sheet": params.sheet, "out_csv": str(out_path), "rows": nrows, "cols": ncols}
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: excel_export_sheet_csv 失败：{type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run()
