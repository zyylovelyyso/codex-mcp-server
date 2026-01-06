# codex-mcp-server
自建的 MCP 服务（MCP Servers）。

# MCP Servers（Web/Excel + File Search + Literature + OpenAI + VizKB）

本仓库提供 5 个本地 stdio MCP 服务器，面向 Codex CLI（或任何支持 MCP 的客户端）：
- `web_excel_mcp.py`：网页抓取 + Excel 读取/导出
- `file_search_mcp.py`：本地文件查找/读取/grep
- `literature_mcp.py`：文献检索与抓取（arXiv/Crossref/DOI→BibTeX）+ 导入 Scholar/CNKI 导出引用
- `openai_mcp.py`：通过 OpenAI 兼容 API 调用 GPT（问答/补丁审查），用于 iFlow 等“模型编排”场景
- `vizkb_mcp.py`：可视化知识库 VizKB（recipe/sources 检索、语义检索、推荐、CSV 预检）

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
- `--vizkb-root /abs/path`：VizKB 根目录（即 `ai-viz-kb/` 文件夹；可选）
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

[mcp_servers.openai]
command = "/abs/path/to/repo/.venv/bin/python"
args = ["/abs/path/to/repo/openai_mcp.py"]

[mcp_servers.vizkb]
command = "/abs/path/to/repo/.venv/bin/python"
args = ["/abs/path/to/repo/vizkb_mcp.py"]
[mcp_servers.vizkb.env]
VIZKB_ROOT = "/abs/path/to/ai-viz-kb"
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

### openai_mcp.py（iFlow：GLM↔GPT 自动协作）

用途：让 iFlow 内置 GLM 在执行过程中“遇到疑问就问 GPT、产出补丁先让 GPT 审查”，再由 iFlow 按 `approve` 分支自动循环。

工具：
- `openai_chat`：通用问答/指导
- `openai_review_patch`：对 unified diff patch 做结构化审查（返回 JSON：approve/must_fix/validation_steps）

环境变量（推荐在 iFlow 的 MCP env 中设置，不要写进流程文本）：
- `OPENAI_MCP_BACKEND`：可选，`http` / `codex` / `auto`（默认 `http`）
  - `http`：直连 OpenAI 兼容 API（需要 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL/OPENAI_MODEL`）
  - `codex`：通过本机 `codex exec` 转发（适合“需要签名/非标准网关”的中转；优先配 `CODEX_HOME`）
  - `auto`：先走 `http`，遇到典型“签名/网关不兼容”错误再回退 `codex`
- `OPENAI_API_KEY`：`http/auto` 模式需要（也可不填，改用 `CODEX_HOME/auth.json` 读取）
- `OPENAI_BASE_URL`：可选（`http/auto` 模式；也可不填，改用 `CODEX_HOME/config.toml` 读取）
- `OPENAI_MODEL`：可选（也可不填，改用 `CODEX_HOME/config.toml` 读取）
- `CODEX_HOME`：可选（默认 `~/.codex`；`codex` 模式建议显式设置）
- `CODEX_CLI`：可选，指定 `codex` 可执行文件路径（默认直接调用 `codex`）

iFlow 配置（stdio，command+args）示例：
- 传输类型：`stdio`
- command：`python3`（或你 venv 的 python 可执行文件）
- args：`/abs/path/to/repo/openai_mcp.py`
- env（可选）：
  - `OPENAI_MCP_BACKEND=codex`（若你的中转对直连 OpenAI HTTP 不兼容，推荐）
  - `CODEX_HOME=/abs/path/to/.codex`
  - `OPENAI_API_KEY=...`
  - `OPENAI_BASE_URL=.../v1`（若你的中转不以 `/v1` 结尾也没关系，脚本会自动补 `/v1`）
  - `OPENAI_MODEL=gpt-5.2`（示例）

GLM 节点建议协议（最小闭环）：
1) GLM 先产出 unified diff patch（不直接落盘）  
2) 调用 `openai_review_patch(goal,constraints,patch,context)`  
3) `approve=false`：按 `must_fix` 逐条返工再审；`approve=true`：再落盘/跑验证  
4) 遇到任何歧义/拿不准：调用 `openai_chat` 获取指导后再继续

备注（参数包装）：
- 本仓库使用 FastMCP，工具签名形如 `tool(params: XxxInput)`；部分 MCP 客户端会要求传参时包一层：
  - `{"params": {...}}`
- iFlow 通常会把字段直接映射到表单/JSON，不需要你手动关心；若你用脚本自测 MCP 协议时再参考这一点。

### vizkb_mcp.py（VizKB：可视化知识库接口）

用途：把本地 `ai-viz-kb/`（recipes + sources + 语义检索）暴露为 MCP 工具，便于 Codex/AI 直接调用：
- 检索：`vizkb_search`（关键词+同义词扩展）
- 语义检索：`vizkb_semantic_search`（TF‑IDF hybrid）
- 场景推荐：`vizkb_recommend`
- 数据预检：`vizkb_validate_csv`
- 取 recipe 路径/说明：`vizkb_get_recipe`

配置要点：
- 必须设置 `VIZKB_ROOT` 指向 VizKB 根目录（也就是 `ai-viz-kb/` 文件夹）。
- 可选设置 `VIZKB_PYTHON` 指向能运行 VizKB 依赖的 Python（默认用启动 MCP server 的解释器）。
