#!/usr/bin/env bash
set -euo pipefail

REPO_URL_DEFAULT="https://github.com/zyylovelyyso/codex-mcp-server.git"
INSTALL_DIR_DEFAULT="${HOME}/.local/share/codex-mcp-server"

PROJECT_ROOT=""
LITERATURE_DIR=""
CODEX_HOME_DIR="${CODEX_HOME:-${HOME}/.codex}"
INSTALL_DIR="${INSTALL_DIR_DEFAULT}"
REPO_URL="${REPO_URL_DEFAULT}"

usage() {
  cat <<'EOF'
install.sh - Install codex-mcp-server and update Codex CLI config.toml

Usage:
  ./install.sh --project-root /abs/path/to/project [--literature-dir /abs/path] [--install-dir /abs/path] [--codex-home /abs/path] [--repo-url URL]

Notes:
  - This script clones/updates the repo, creates a venv, installs requirements, and upserts MCP sections into:
      $CODEX_HOME/config.toml   (default: ~/.codex/config.toml)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="${2:-}"; shift 2 ;;
    --literature-dir)
      LITERATURE_DIR="${2:-}"; shift 2 ;;
    --install-dir)
      INSTALL_DIR="${2:-}"; shift 2 ;;
    --codex-home)
      CODEX_HOME_DIR="${2:-}"; shift 2 ;;
    --repo-url)
      REPO_URL="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage; exit 2 ;;
  esac
done

if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "Missing required: --project-root" >&2
  usage
  exit 2
fi

PROJECT_ROOT="$(python3 - <<PY
from pathlib import Path
print(Path(r'''${PROJECT_ROOT}''').expanduser().resolve())
PY
)"

if [[ -z "${LITERATURE_DIR}" ]]; then
  if [[ -d "${PROJECT_ROOT}/02-文献与资料" ]]; then
    LITERATURE_DIR="${PROJECT_ROOT}/02-文献与资料"
  else
    LITERATURE_DIR="${PROJECT_ROOT}"
  fi
fi

LITERATURE_DIR="$(python3 - <<PY
from pathlib import Path
print(Path(r'''${LITERATURE_DIR}''').expanduser().resolve())
PY
)"

mkdir -p "${INSTALL_DIR}"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "[1/4] Updating repo in ${INSTALL_DIR}"
  git -C "${INSTALL_DIR}" pull --ff-only
else
  echo "[1/4] Cloning repo to ${INSTALL_DIR}"
  git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

echo "[2/4] Creating venv and installing requirements"
python3 -m venv "${INSTALL_DIR}/.venv"
"${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

echo "[3/4] Updating Codex config.toml"
mkdir -p "${CODEX_HOME_DIR}"
CONFIG_PATH="${CODEX_HOME_DIR}/config.toml"
touch "${CONFIG_PATH}"

"${INSTALL_DIR}/.venv/bin/python" "${INSTALL_DIR}/scripts/setup_codex_config.py" \
  --config "${CONFIG_PATH}" \
  --repo-dir "${INSTALL_DIR}" \
  --project-root "${PROJECT_ROOT}" \
  --literature-dir "${LITERATURE_DIR}"

echo "[4/4] Done"
echo "Restart Codex CLI to load new MCP servers."
echo "Config updated: ${CONFIG_PATH}"

