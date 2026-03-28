#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

VERSION_SUFFIX="${1:-}"
OS_TAG="${2:-}"
APP_BASE_NAME="${LIVEFT_APP_BASE_NAME:-LiveFT}"
APP_NAME="${APP_BASE_NAME}"

if [ -n "${VERSION_SUFFIX}" ]; then
  APP_NAME="${APP_NAME}-${VERSION_SUFFIX}"
fi

if [ -n "${OS_TAG}" ]; then
  APP_NAME="${APP_NAME}-${OS_TAG}"
fi

export LIVEFT_APP_NAME="${APP_NAME}"
export LIVEFT_BUNDLE_ID="${LIVEFT_BUNDLE_ID:-nl.stack.liveft}"
export LIVEFT_CAMERA_USAGE="${LIVEFT_CAMERA_USAGE:-LiveFT needs camera access to capture the live image used for Fourier-transform visualization.}"
export LIVEFT_CODESIGN_IDENTITY="${LIVEFT_CODESIGN_IDENTITY:-}"
export LIVEFT_ENTITLEMENTS_FILE="${LIVEFT_ENTITLEMENTS_FILE:-CodeSigning/entitlements.xml}"
export LIVEFT_ICON_PATH="${LIVEFT_ICON_PATH:-}"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-${REPO_ROOT}/.pyinstaller}"

if [ -n "${LIVEFT_PYTHON:-}" ]; then
  PYTHON_BIN="${LIVEFT_PYTHON}"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No Python interpreter found. Set LIVEFT_PYTHON or create .venv/bin/python." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${PYINSTALLER_CONFIG_DIR}"
"${PYTHON_BIN}" -m PyInstaller --clean -y LiveFT.spec
