# codex-mcp-server
自建的 MCP 服务（MCP Servers）。

# MCP Servers（Web/Excel + File Search + Literature）

本仓库提供 3 个本地 stdio MCP 服务器，面向 Codex CLI（或任何支持 MCP 的客户端）：
- `web_excel_mcp.py`：网页抓取 + Excel 读取/导出
- `file_search_mcp.py`：本地文件查找/读取/grep
- `literature_mcp.py`：文献检索与抓取（arXiv/Crossref/DOI→BibTeX）+ 导入 Scholar/CNKI 导出引用

## 1) 安装依赖（建议用 venv）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 0) 一键安装并写入 Codex 配置（推荐）

在 WSL/Linux 下：
```bash
chmod +x install.sh
./install.sh --project-root /abs/path/to/your/project
```

可选参数：
- `--literature-dir /abs/path`：文献导入/下载目录（默认优先用 `project/02-文献与资料`）
- `--codex-home /abs/path`：Codex 配置目录（默认 `~/.codex` 或环境变量 `CODEX_HOME`）
- `--install-dir /abs/path`：仓库安装目录（默认 `~/.local/share/codex-mcp-server`）

## 2) 本地启动（用于手动调试）

```bash
MCP_ALLOWED_ROOT=/path/to/allowed/root python3 web_excel_mcp.py
```

说明：`MCP_ALLOWED_ROOT` 用来限制可读写的文件根目录（默认是启动时的当前目录）。

## 3) 注册到 Codex（一次即可）

两种方式任选其一：

### 方式 A：写 `~/.codex/config.toml`（推荐）

示例：
```toml
[mcp_servers.web_excel]
command = "/abs/path/to/repo/.venv/bin/python"
args = ["/abs/path/to/repo/web_excel_mcp.py"]
[mcp_servers.web_excel.env]
MCP_ALLOWED_ROOT = "/abs/path/to/project"

[mcp_servers.files]
command = "/abs/path/to/repo/.venv/bin/python"
args = ["/abs/path/to/repo/file_search_mcp.py"]
[mcp_servers.files.env]
MCP_ALLOWED_ROOTS = "/abs/path/to/project"

[mcp_servers.literature]
command = "/abs/path/to/repo/.venv/bin/python"
args = ["/abs/path/to/repo/literature_mcp.py"]
[mcp_servers.literature.env]
MCP_ALLOWED_ROOT = "/abs/path/to/literature_dir"
```

### 方式 B：使用 `codex mcp add`（如果你的 CLI 支持）
```bash
export CODEX_HOME=~/.codex
codex mcp add web_excel --env MCP_ALLOWED_ROOT=/abs/path/to/project -- python3 /abs/path/to/repo/web_excel_mcp.py
codex mcp list
```

## 4) 工具列表（服务器内提供）

- `web_fetch`：抓取网页/下载文件（可保存到项目目录）
- `excel_list_sheets`：列出工作表
- `excel_read_sheet_preview`：预览工作表前 n 行
- `excel_profile_sheet`：列结构/缺失/示例值统计
- `excel_export_sheet_csv`：导出工作表为 CSV
## 5) 其它服务器

### file_search_mcp.py

- 主要工具：`fs_find` / `fs_read` / `fs_grep`
- 权限：通过环境变量 `MCP_ALLOWED_ROOTS`（可多个）限制可访问目录

### literature_mcp.py

- 主要工具：
  - `lit_search_arxiv`：检索 arXiv
  - `lit_search_crossref`：检索 Crossref（DOI/元信息）
  - `lit_get_bibtex`：DOI → BibTeX
  - `lit_download`：下载 PDF/文件到本地
  - `lit_import_bibtex_text` / `lit_import_bibtex_file` / `lit_import_ris_file`：导入引用（适配 Google Scholar / CNKI “导出引用”）
- CNKI / Google Scholar：建议在网页端导出 BibTeX/RIS，再用上述导入工具纳入可复核的引用流程，而不是直接自动爬网页。
