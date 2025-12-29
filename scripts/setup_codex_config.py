#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


SECTION_HEADER_RE = re.compile(r"(?m)^\[([^\]]+)\]\s*$")


def _upsert_section(text: str, section: str, body_lines: list[str]) -> str:
    header_line = f"[{section}]"
    body = "\n".join(body_lines).rstrip() + "\n"

    m = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
    if not m:
        if text and not text.endswith("\n"):
            text += "\n"
        if text and not text.endswith("\n\n"):
            text += "\n"
        return text + header_line + "\n" + body

    start = m.start()
    after = text[m.end() :]
    m2 = SECTION_HEADER_RE.search(after)
    end = m.end() + (m2.start() if m2 else len(after))

    before = text[:start]
    after_rest = text[end:]
    if before and not before.endswith("\n\n"):
        before = before.rstrip("\n") + "\n\n"
    replacement = header_line + "\n" + body
    if after_rest and not replacement.endswith("\n\n"):
        replacement = replacement.rstrip("\n") + "\n\n"
    return before + replacement + after_rest.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert MCP server sections into Codex config.toml")
    parser.add_argument("--config", required=True, help="Path to ~/.codex/config.toml (or other CODEX_HOME)")
    parser.add_argument("--repo-dir", required=True, help="Path to cloned codex-mcp-server repo")
    parser.add_argument("--project-root", required=True, help="MCP_ALLOWED_ROOT / MCP_ALLOWED_ROOTS value")
    parser.add_argument("--literature-dir", required=True, help="MCP_ALLOWED_ROOT for literature downloads/imports")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    literature_dir = Path(args.literature_dir).expanduser().resolve()

    python_cmd = repo_dir / ".venv" / "bin" / "python"
    if not python_cmd.exists():
        python_cmd = Path("python3")

    web_excel = repo_dir / "web_excel_mcp.py"
    files = repo_dir / "file_search_mcp.py"
    literature = repo_dir / "literature_mcp.py"

    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""

    text = _upsert_section(
        text,
        "mcp_servers.web_excel",
        [f'command = "{python_cmd}"', f'args = ["{web_excel}"]'],
    )
    text = _upsert_section(
        text,
        "mcp_servers.web_excel.env",
        [f'MCP_ALLOWED_ROOT = "{project_root}"'],
    )

    text = _upsert_section(
        text,
        "mcp_servers.files",
        [f'command = "{python_cmd}"', f'args = ["{files}"]'],
    )
    text = _upsert_section(
        text,
        "mcp_servers.files.env",
        [
            "# Multiple roots are separated by OS path separator (: on Linux/WSL, ; on Windows)",
            f'MCP_ALLOWED_ROOTS = "{project_root}"',
        ],
    )

    text = _upsert_section(
        text,
        "mcp_servers.literature",
        [f'command = "{python_cmd}"', f'args = ["{literature}"]'],
    )
    text = _upsert_section(
        text,
        "mcp_servers.literature.env",
        [f'MCP_ALLOWED_ROOT = "{literature_dir}"'],
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

