#!/usr/bin/env python3
"""
openai_mcp: 通过 OpenAI 兼容 API 调用 GPT 的本地 MCP 服务器（stdio）。

设计目标（KISS）：
- 提供两个最小但可组合的工具：
  1) openai_chat：通用问答/指导（system + user）
  2) openai_review_patch：对 unified diff patch 做结构化审查（JSON：approve/must_fix/...）
- 不自动读取/上传仓库文件；只发送调用方显式传入的文本，避免“隐式数据外发”。
- 默认优先复用本机 Codex 配置（~/.codex），减少 iFlow 配置成本：
  - OPENAI_API_KEY：优先读环境变量，否则尝试从 `CODEX_HOME/auth.json` 读取
  - OPENAI_BASE_URL / OPENAI_MODEL：优先读环境变量，否则尝试从 `CODEX_HOME/config.toml` 读取
- 后端选择：
  - OPENAI_MCP_BACKEND=http（默认）：直连 OpenAI 兼容 HTTP API
  - OPENAI_MCP_BACKEND=codex：通过本机 `codex exec` 转发（适配“签名/非标准网关”）
  - OPENAI_MCP_BACKEND=auto：先 http，典型签名错误时回退 codex
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover (py3.10)
    tomllib = None  # type: ignore[assignment]

try:  # py3.10 常用
    import tomli  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    tomli = None  # type: ignore[assignment]


mcp = FastMCP("openai_mcp")


def _load_backend() -> str:
    """
    选择“如何调用 GPT”的后端。

    - http:   直接走 OpenAI 兼容 HTTP API（默认）
    - codex:  通过本机 `codex exec` 转发（适配某些需要签名/非标准网关的中转）
    - auto:   先 http，遇到典型“签名/网关不兼容”错误则回退到 codex
    """

    raw = (os.environ.get("OPENAI_MCP_BACKEND") or "").strip().lower()
    if not raw:
        return "http"
    if raw in {"http", "openai", "direct"}:
        return "http"
    if raw in {"codex", "codex-cli", "codex_cli", "cli"}:
        return "codex"
    if raw in {"auto"}:
        return "auto"
    raise ValueError("OPENAI_MCP_BACKEND 仅支持 http / codex / auto")


def _codex_home() -> Path:
    raw = (os.environ.get("CODEX_HOME") or os.environ.get("CODEX_CONFIG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def _load_openai_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key

    auth_path = _codex_home() / "auth.json"
    if auth_path.exists():
        try:
            obj = json.loads(auth_path.read_text(encoding="utf-8"))
            key2 = (obj.get("OPENAI_API_KEY") or "").strip()
            if key2:
                return key2
        except Exception:
            pass

    raise ValueError(
        "未检测到 OPENAI_API_KEY。请在 iFlow 的 MCP 环境变量中设置 OPENAI_API_KEY，"
        "或确保存在可读的 CODEX_HOME/auth.json（包含 OPENAI_API_KEY）。"
    )


def _load_codex_defaults() -> tuple[str | None, str | None]:
    """
    尝试从 `CODEX_HOME/config.toml` 读取 (base_url, model)。
    不保证一定存在；读取失败则返回 (None, None)。
    """

    cfg_path = _codex_home() / "config.toml"
    if not cfg_path.exists():
        return None, None

    text = cfg_path.read_text(encoding="utf-8", errors="replace")

    # 1) 优先用 tomllib/tomli（若可用）
    toml_loader = tomllib or tomli
    if toml_loader is not None:
        try:
            cfg = toml_loader.loads(text)  # type: ignore[attr-defined]
            model = cfg.get("model")
            provider = cfg.get("model_provider")
            base_url: str | None = None

            providers = cfg.get("model_providers") or {}
            if isinstance(provider, str) and provider and isinstance(providers, dict):
                p = providers.get(provider) or {}
                if isinstance(p, dict):
                    base_url = p.get("base_url")

            return (base_url if isinstance(base_url, str) else None, model if isinstance(model, str) else None)
        except Exception:
            pass

    # 2) 最小手写解析（仅解析 model_provider/model/base_url，避免额外依赖）
    model_provider: str | None = None
    model: str | None = None
    base_urls: dict[str, str] = {}
    current_provider: str | None = None

    def _unquote(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            return v[1:-1]
        return v

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = re.match(r"^\[(.+?)\]\s*$", line)
        if m:
            sec = m.group(1).strip()
            m2 = re.match(r"^model_providers\.([A-Za-z0-9_-]+)$", sec)
            current_provider = m2.group(1) if m2 else None
            continue

        m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1), _unquote(m.group(2))

        if current_provider is None:
            if k == "model_provider":
                model_provider = v
            elif k == "model":
                model = v
        else:
            if k == "base_url":
                base_urls[current_provider] = v

    base_url = base_urls.get(model_provider or "")
    return (base_url, model)


def _load_openai_base_url() -> str:
    raw = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if raw:
        return raw

    base_url, _model = _load_codex_defaults()
    if base_url:
        return base_url

    # 最保守的兜底：官方 OpenAI
    return "https://api.openai.com/v1"


def _load_openai_model() -> str:
    raw = (os.environ.get("OPENAI_MODEL") or "").strip()
    if raw:
        return raw

    _base_url, model = _load_codex_defaults()
    if model:
        return model

    # 兜底：尽量贴近本项目常用（你也可以在 iFlow 环境里覆盖）
    return "gpt-5.2"


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _candidate_responses_urls(base_url: str) -> list[str]:
    """
    生成候选 `responses` 端点 URL（兼容不同“中转/网关”的 base_url 习惯）。

    经验上常见三类：
    - 直接给到 v1：   https://api.xxx.com/v1
    - 给到 openai 根： https://api.xxx.com/openai   （期望 /openai/responses）
    - 给到 host 根：  https://api.xxx.com          （期望 /v1/responses）
    """

    base = base_url.strip().rstrip("/")
    candidates = [_join_url(base, "responses")]

    # 若 base 不以 /v1 结尾，则再尝试 base + /v1/responses（覆盖“给到 host 根”的写法）
    if not re.search(r"/v1$", base):
        candidates.append(_join_url(base + "/v1", "responses"))

    # 去重（保序）
    seen: set[str] = set()
    uniq: list[str] = []
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def _find_codex_cli() -> str:
    """
    寻找 codex CLI 可执行文件。
    允许用环境变量 `CODEX_CLI` 覆盖（例如给绝对路径）。
    """

    raw = (os.environ.get("CODEX_CLI") or "codex").strip()
    if not raw:
        raw = "codex"

    # 绝对/相对路径：直接使用；否则尝试 PATH 查找
    if os.path.sep in raw or raw.startswith("."):
        return raw

    found = shutil.which(raw)
    return found or raw


def _run_codex_exec_sync(*, prompt: str, model: str, timeout_seconds: float) -> str:
    """
    使用 `codex exec` 做一次非交互调用，并返回“最后一条消息”的纯文本。

    注意：prompt 可能很长（diff/上下文），必须通过 stdin 传递，避免命令行长度上限。
    """

    codex_cli = _find_codex_cli()

    fd, out_path = tempfile.mkstemp(prefix="openai_mcp_codex_", suffix=".txt")
    os.close(fd)

    try:
        cmd = [
            codex_cli,
            "exec",
            "-s",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            out_path,
            "-m",
            model,
            "-",
        ]

        env = dict(os.environ)
        env.setdefault("CODEX_HOME", str(_codex_home()))

        try:
            res = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env=env,
            )
        except FileNotFoundError as e:
            raise ValueError(
                "未找到 codex CLI。请确认已安装 `codex` 并在 PATH 中，或设置环境变量 CODEX_CLI 指向可执行文件。"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"codex exec 超时（{timeout_seconds}s）。可适当增大 timeout_seconds。") from e

        if res.returncode != 0:
            stdout = (res.stdout or "").strip()
            stderr = (res.stderr or "").strip()
            msg = "\n".join(x for x in [stdout, stderr] if x)
            if len(msg) > 2000:
                msg = msg[:2000] + "…(truncated)"
            raise ValueError(f"codex exec 失败（exit={res.returncode}）。{msg}")

        try:
            return Path(out_path).read_text(encoding="utf-8", errors="replace").strip()
        except FileNotFoundError as e:
            raise ValueError("codex exec 未生成 output-last-message 文件（可能被外部清理或权限不足）。") from e
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass


async def _codex_exec_text(*, prompt: str, model: str, timeout_seconds: float) -> str:
    # FastMCP 的工具函数是 async；子进程调用用 to_thread 避免阻塞事件循环。
    return await asyncio.to_thread(_run_codex_exec_sync, prompt=prompt, model=model, timeout_seconds=timeout_seconds)


def _extract_text_from_openai_response(data: dict[str, Any]) -> str:
    # 1) responses API: some gateways may expose output_text directly
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    # 2) responses API: parse output -> message -> content -> output_text
    out = data.get("output")
    if isinstance(out, list):
        chunks: list[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        if chunks:
            return "".join(chunks).strip()

    # 3) chat completions fallback shape
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]

    return ""


async def _post_openai_responses(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    urls = _candidate_responses_urls(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
        async def _post_streaming(url: str) -> dict[str, Any]:
            """
            兼容部分中转仅支持 stream 的情况：读取 SSE 并拼接 output_text。

            说明：这里不追求完整还原响应对象，只要能稳定得到文本即可满足 openai_chat/openai_review_patch 的需求。
            """

            payload2 = dict(payload)
            payload2["stream"] = True

            headers2 = dict(headers)
            headers2["Accept"] = "text/event-stream"

            async with client.stream("POST", url, headers=headers2, json=payload2) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    if len(body) > 2000:
                        body = body[:2000] + "…(truncated)"
                    raise ValueError(f"OpenAI 请求失败：HTTP {resp.status_code}（url={url}）: {body}")

                chunks: list[str] = []
                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    data_line = line
                    if data_line.startswith("data:"):
                        data_line = data_line[len("data:") :].strip()

                    if data_line == "[DONE]":
                        break

                    if not data_line:
                        continue

                    try:
                        obj = json.loads(data_line)
                    except Exception:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    # Responses API streaming: {"type":"response.output_text.delta","delta":"..."}
                    if isinstance(obj.get("delta"), str):
                        chunks.append(obj["delta"])
                        continue

                    # ChatCompletions streaming: {"choices":[{"delta":{"content":"..."}}]}
                    choices = obj.get("choices")
                    if isinstance(choices, list) and choices:
                        c0 = choices[0]
                        if isinstance(c0, dict):
                            delta = c0.get("delta")
                            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                chunks.append(delta["content"])
                                continue

                return {"output_text": "".join(chunks).strip(), "id": None, "model": payload.get("model"), "usage": None}

        for url in urls:
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as e:
                last_error = e
                continue
            except httpx.RequestError as e:
                last_error = e
                continue

            # 对于“base_url 需补 /v1”的情况，常见表现是 404；仅在多候选时继续尝试
            if resp.status_code == 404 and len(urls) > 1:
                last_error = ValueError(f"HTTP 404: {url}")
                continue

            if resp.status_code >= 400:
                body = resp.text
                # 一些中转只支持 stream，非 stream 会返回 400 + "only support stream"
                if resp.status_code == 400 and "only support stream" in body.lower():
                    return await _post_streaming(url)

                if len(body) > 2000:
                    body = body[:2000] + "…(truncated)"
                raise ValueError(f"OpenAI 请求失败：HTTP {resp.status_code}（url={url}）: {body}")

            return resp.json()

    if isinstance(last_error, httpx.TimeoutException):
        raise ValueError(
            f"OpenAI 请求超时：{type(last_error).__name__}（urls={urls}）。"
            "请检查 OPENAI_BASE_URL 是否可达、网络/代理是否正常，并适当增大 timeout_seconds。"
        ) from last_error
    if isinstance(last_error, httpx.RequestError):
        raise ValueError(
            f"OpenAI 请求失败：{type(last_error).__name__}（urls={urls}）。"
            "请检查 OPENAI_BASE_URL 是否正确、网络是否可用。"
        ) from last_error
    raise ValueError(f"OpenAI 请求失败：未找到可用 endpoints（urls={urls}）。最后错误：{last_error!r}")


def _strip_code_fence(s: str) -> str:
    t = s.strip()
    # ```json ... ```  / ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_first_json_object(s: str) -> dict[str, Any]:
    t = _strip_code_fence(s)
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("未找到 JSON 对象边界")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层不是对象")
    return obj


class OpenAIChatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    system: str | None = Field(default=None, description="可选：system 提示词", max_length=20_000)
    user: str = Field(..., description="user 输入", min_length=1, max_length=200_000)
    model: str | None = Field(default=None, description="可选：覆盖 OPENAI_MODEL", max_length=200)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度")
    max_output_tokens: int = Field(default=1024, ge=1, le=8192, description="最多输出 token")
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=180.0, description="请求超时（秒）")
    base_url: str | None = Field(default=None, description="可选：覆盖 OPENAI_BASE_URL", max_length=2000)


@mcp.tool(
    name="openai_chat",
    annotations={
        "title": "调用 OpenAI（通用问答）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def openai_chat(params: OpenAIChatInput) -> str:
    model = (params.model or _load_openai_model()).strip()

    backend = _load_backend()

    # codex 后端：通过 codex exec 转发（适配非标准网关/签名网关）
    if backend == "codex":
        prompt = f"{params.user}".strip()
        if params.system:
            prompt = f"System:\n{params.system}\n\nUser:\n{params.user}".strip()
        text = await _codex_exec_text(prompt=prompt, model=model, timeout_seconds=params.timeout_seconds)
        out = {"text": text, "model": model, "request_id": None, "usage": None}
        return json.dumps(out, ensure_ascii=False, indent=2)

    # http / auto：优先走 OpenAI 兼容 API
    base_url = (params.base_url or _load_openai_base_url()).strip()
    try:
        api_key = _load_openai_api_key()
        messages: list[dict[str, Any]] = []
        if params.system:
            messages.append({"role": "system", "content": params.system})
        messages.append({"role": "user", "content": params.user})

        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "temperature": params.temperature,
            "max_output_tokens": params.max_output_tokens,
        }

        data = await _post_openai_responses(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout_seconds=params.timeout_seconds,
        )

        text = _extract_text_from_openai_response(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        out = {"text": text, "model": data.get("model") or model, "request_id": data.get("id"), "usage": usage}
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as e:
        # auto：遇到典型“签名/网关不兼容”错误时回退到 codex
        if backend == "auto":
            msg = str(e)
            if ("not codex request" in msg.lower()) or ("签名" in msg):
                prompt = f"{params.user}".strip()
                if params.system:
                    prompt = f"System:\n{params.system}\n\nUser:\n{params.user}".strip()
                text = await _codex_exec_text(prompt=prompt, model=model, timeout_seconds=params.timeout_seconds)
                out = {"text": text, "model": model, "request_id": None, "usage": None}
                return json.dumps(out, ensure_ascii=False, indent=2)
        raise


class OpenAIReviewPatchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    goal: str = Field(..., description="本次改动目标", min_length=1, max_length=10_000)
    constraints: str = Field(default="", description="约束/边界（KISS/中文注释/不可删目录等）", max_length=20_000)
    patch: str = Field(..., description="unified diff patch 文本", min_length=1, max_length=300_000)
    context: str | None = Field(default=None, description="可选：补充上下文（关键文件摘要/指标结果等）", max_length=200_000)
    model: str | None = Field(default=None, description="可选：覆盖 OPENAI_MODEL", max_length=200)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=180.0, description="请求超时（秒）")
    base_url: str | None = Field(default=None, description="可选：覆盖 OPENAI_BASE_URL", max_length=2000)


_REVIEW_SYSTEM = """你是严格的代码审查与逻辑审查助手。
你必须只输出 JSON（不要 Markdown、不要解释文字、不要代码块）。
输出 JSON 必须包含以下字段：
- approve: boolean
- must_fix: string[]
- should_fix: string[]
- validation_steps: string[]
- notes: string（可选；建议简短）

判定规则：
- 发现会导致错误、违反约束、口径不一致、明显不合理的风险：approve=false，并把原因写入 must_fix。
- approve=true 时，must_fix 必须为空数组。
- validation_steps 给出“最小可执行”的验证命令清单（例如：py_compile、单元测试、latexmk 等）。
"""


@mcp.tool(
    name="openai_review_patch",
    annotations={
        "title": "审查补丁（结构化 JSON 裁决）",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def openai_review_patch(params: OpenAIReviewPatchInput) -> str:
    model = (params.model or _load_openai_model()).strip()
    backend = _load_backend()

    user_parts = [
        f"Goal:\n{params.goal}",
        f"Constraints:\n{params.constraints or '(none)'}",
        "Patch (unified diff):\n" + params.patch,
    ]
    if params.context:
        user_parts.append("Context:\n" + params.context)
    user_parts.append("只输出 JSON。")

    # codex 后端：用单段 prompt 强制 JSON 输出
    if backend == "codex":
        prompt = _REVIEW_SYSTEM + "\n\n" + "\n\n".join(user_parts)
        raw_text = (await _codex_exec_text(prompt=prompt, model=model, timeout_seconds=params.timeout_seconds)).strip()
        usage = None
        request_id = None
    else:
        base_url = (params.base_url or _load_openai_base_url()).strip()
        try:
            api_key = _load_openai_api_key()
            payload: dict[str, Any] = {
                "model": model,
                "input": [
                    {"role": "system", "content": _REVIEW_SYSTEM},
                    {"role": "user", "content": "\n\n".join(user_parts)},
                ],
                "temperature": 0.0,
                "max_output_tokens": 1200,
            }

            data = await _post_openai_responses(
                base_url=base_url,
                api_key=api_key,
                payload=payload,
                timeout_seconds=params.timeout_seconds,
            )

            raw_text = _extract_text_from_openai_response(data).strip()
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            request_id = data.get("id")
        except Exception as e:
            if backend == "auto":
                msg = str(e)
                if ("not codex request" in msg.lower()) or ("签名" in msg):
                    prompt = _REVIEW_SYSTEM + "\n\n" + "\n\n".join(user_parts)
                    raw_text = (
                        await _codex_exec_text(prompt=prompt, model=model, timeout_seconds=params.timeout_seconds)
                    ).strip()
                    usage = None
                    request_id = None
                else:
                    raise
            else:
                raise

    try:
        obj = _parse_first_json_object(raw_text)
        # 轻量归一化（保证字段存在、类型合理）
        approve = bool(obj.get("approve"))
        must_fix = obj.get("must_fix") if isinstance(obj.get("must_fix"), list) else []
        should_fix = obj.get("should_fix") if isinstance(obj.get("should_fix"), list) else []
        validation_steps = obj.get("validation_steps") if isinstance(obj.get("validation_steps"), list) else []
        notes = obj.get("notes") if isinstance(obj.get("notes"), str) else ""

        if approve and must_fix:
            approve = False
            must_fix = ["approve=true 但 must_fix 非空；请修正输出为一致的 JSON。"] + [str(x) for x in must_fix]

        normalized = {
            "approve": approve,
            "must_fix": [str(x) for x in must_fix],
            "should_fix": [str(x) for x in should_fix],
            "validation_steps": [str(x) for x in validation_steps],
            "notes": notes,
            "model": model,
            "request_id": request_id,
            "usage": usage,
        }
        return json.dumps(normalized, ensure_ascii=False, indent=2)
    except Exception as e:
        # 按 spec：返回 parse error + raw text，便于 iFlow 触发“重试/强化提示词/缩短上下文”
        truncated = raw_text if len(raw_text) <= 6000 else (raw_text[:6000] + "…(truncated)")
        fallback = {
            "approve": False,
            "must_fix": [
                "模型输出无法解析为 JSON；请重试（建议：缩短 patch/context 或强化提示词）。",
                f"parse_error: {type(e).__name__}: {e}",
            ],
            "should_fix": [],
            "validation_steps": [],
            "notes": truncated,
            "model": model,
            "request_id": request_id,
            "usage": usage,
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
