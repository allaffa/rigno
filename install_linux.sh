#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${RIGNO_VENV_DIR:-${PROJECT_DIR}/.venv}"
PYTHON_BIN="${RIGNO_PYTHON:-python3}"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./install_linux.sh

Creates a Python virtual environment and installs requirements.txt.

Optional environment variables:
  RIGNO_PYTHON     Python executable to use (default: python3)
  RIGNO_VENV_DIR   Virtual-environment path (default: <repo>/.venv)
EOF
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: install_linux.sh supports Linux only." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' was not found." >&2
  echo "Install Python 3.10 or newer, including the venv module." >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Error: ${REQUIREMENTS_FILE} does not exist." >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Error: Python 3.10 or newer is required; found ${PYTHON_VERSION}." >&2
  exit 1
fi

echo "Creating virtual environment with ${PYTHON_BIN} (${PYTHON_VERSION}): ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install --requirement "${REQUIREMENTS_FILE}"

"${VENV_PYTHON}" - <<'PY'
import torch
import torch_geometric

print(f"PyTorch {torch.__version__}")
print(f"PyTorch Geometric {torch_geometric.__version__}")
PY

cat <<EOF

Installation complete.
Activate the environment with:
  source "${VENV_DIR}/bin/activate"

Run the correctness tests with:
  python -m pytest -q tests
EOF
