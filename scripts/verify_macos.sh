#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET_INPUT="${1:-dist}"
EXPECT_NOTARIZED="${LIVEFT_EXPECT_NOTARIZED:-0}"
VERIFIED_ANY=0

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

verify_app() {
  app_path="$1"
  printf 'Verifying app bundle: %s\n' "${app_path}"
  codesign --verify --deep --strict "${app_path}"
  if [ "${EXPECT_NOTARIZED}" = "1" ]; then
    spctl -a -vvv --type exec "${app_path}"
    xcrun stapler validate "${app_path}"
  fi
}

verify_dmg() {
  dmg_path="$1"
  printf 'Verifying disk image: %s\n' "${dmg_path}"
  if codesign -dv --verbose=2 "${dmg_path}" >/dev/null 2>&1; then
    codesign --verify "${dmg_path}"
  elif [ "${EXPECT_NOTARIZED}" = "1" ]; then
    printf 'Disk image is not signed: %s\n' "${dmg_path}" >&2
    exit 1
  else
    printf 'Skipping unsigned DMG signature verification: %s\n' "${dmg_path}"
  fi

  if [ "${EXPECT_NOTARIZED}" = "1" ]; then
    spctl -a -vvv --type open "${dmg_path}"
    xcrun stapler validate "${dmg_path}"
  fi
}

verify_target() {
  target_path="$1"

  case "${target_path}" in
    *.app)
      VERIFIED_ANY=1
      verify_app "${target_path}"
      ;;
    *.dmg)
      VERIFIED_ANY=1
      verify_dmg "${target_path}"
      ;;
    *)
      if [ ! -d "${target_path}" ]; then
        printf 'Unsupported verification target: %s\n' "${target_path}" >&2
        exit 1
      fi

      for candidate in "${target_path}"/LiveFT*.app; do
        [ -e "${candidate}" ] || continue
        VERIFIED_ANY=1
        verify_app "${candidate}"
      done

      for candidate in "${target_path}"/LiveFT*.dmg; do
        [ -e "${candidate}" ] || continue
        VERIFIED_ANY=1
        verify_dmg "${candidate}"
      done
      ;;
  esac
}

TARGET_PATH=$(resolve_path "${TARGET_INPUT}")
if [ ! -e "${TARGET_PATH}" ]; then
  printf 'Verification target not found: %s\n' "${TARGET_PATH}" >&2
  exit 1
fi

verify_target "${TARGET_PATH}"

if [ "${VERIFIED_ANY}" -ne 1 ]; then
  printf 'No LiveFT macOS artifacts found under %s\n' "${TARGET_PATH}" >&2
  exit 1
fi
