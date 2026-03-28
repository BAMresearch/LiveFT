#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET_INPUT="${1:-dist}"
CODE_SIGN_IDENTITY="${LIVEFT_CODESIGN_IDENTITY:-}"
ENTITLEMENTS_INPUT="${LIVEFT_ENTITLEMENTS_FILE:-${REPO_ROOT}/CodeSigning/entitlements.xml}"
SIGN_OPTIONS="${LIVEFT_CODESIGN_OPTIONS:-runtime}"
SIGNED_ANY=0

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

sign_app() {
  app_path="$1"
  printf 'Signing app bundle: %s\n' "${app_path}"
  codesign --force --deep --options "${SIGN_OPTIONS}" --entitlements "${ENTITLEMENTS_FILE}" --sign "${CODE_SIGN_IDENTITY}" "${app_path}"
  codesign --verify --deep --strict "${app_path}"
}

sign_dmg() {
  dmg_path="$1"
  printf 'Signing disk image: %s\n' "${dmg_path}"
  codesign --force --sign "${CODE_SIGN_IDENTITY}" "${dmg_path}"
  codesign --verify "${dmg_path}"
}

sign_target() {
  target_path="$1"

  case "${target_path}" in
    *.app)
      SIGNED_ANY=1
      sign_app "${target_path}"
      ;;
    *.dmg)
      SIGNED_ANY=1
      sign_dmg "${target_path}"
      ;;
    *)
      if [ ! -d "${target_path}" ]; then
        printf 'Unsupported signing target: %s\n' "${target_path}" >&2
        exit 1
      fi

      for candidate in "${target_path}"/LiveFT*.app; do
        [ -e "${candidate}" ] || continue
        SIGNED_ANY=1
        sign_app "${candidate}"
      done

      for candidate in "${target_path}"/LiveFT*.dmg; do
        [ -e "${candidate}" ] || continue
        SIGNED_ANY=1
        sign_dmg "${candidate}"
      done
      ;;
  esac
}

if [ -z "${CODE_SIGN_IDENTITY}" ]; then
  printf 'LIVEFT_CODESIGN_IDENTITY is required for macOS signing.\n' >&2
  exit 1
fi

if [ -z "${ENTITLEMENTS_INPUT}" ]; then
  printf 'LIVEFT_ENTITLEMENTS_FILE must not be empty.\n' >&2
  exit 1
fi

ENTITLEMENTS_FILE=$(resolve_path "${ENTITLEMENTS_INPUT}")
if [ ! -f "${ENTITLEMENTS_FILE}" ]; then
  printf 'Entitlements file not found: %s\n' "${ENTITLEMENTS_FILE}" >&2
  exit 1
fi

TARGET_PATH=$(resolve_path "${TARGET_INPUT}")
if [ ! -e "${TARGET_PATH}" ]; then
  printf 'Signing target not found: %s\n' "${TARGET_PATH}" >&2
  exit 1
fi

sign_target "${TARGET_PATH}"

if [ "${SIGNED_ANY}" -ne 1 ]; then
  printf 'No LiveFT macOS artifacts found under %s\n' "${TARGET_PATH}" >&2
  exit 1
fi
