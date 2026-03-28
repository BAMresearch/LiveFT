#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET_INPUT="${1:-dist}"
NOTARIZED_ANY=0

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

notarize_target() {
  target_path="$1"
  printf 'Submitting for notarization: %s\n' "${target_path}"

  case "${NOTARY_MODE}" in
    profile)
      xcrun notarytool submit "${target_path}" --wait --keychain-profile "${LIVEFT_NOTARY_PROFILE}"
      ;;
    api_key)
      xcrun notarytool submit "${target_path}" --wait --key "${LIVEFT_NOTARY_KEY_FILE}" --key-id "${LIVEFT_NOTARY_KEY_ID}" --issuer "${LIVEFT_NOTARY_ISSUER}"
      ;;
    apple_id)
      xcrun notarytool submit "${target_path}" --wait --apple-id "${LIVEFT_NOTARY_APPLE_ID}" --team-id "${LIVEFT_NOTARY_TEAM_ID}" --password "${LIVEFT_NOTARY_PASSWORD}"
      ;;
  esac

  xcrun stapler staple "${target_path}"
  xcrun stapler validate "${target_path}"
}

if ! command -v xcrun >/dev/null 2>&1; then
  printf 'xcrun is required for notarization.\n' >&2
  exit 1
fi

NOTARY_MODE=""
if [ -n "${LIVEFT_NOTARY_PROFILE:-}" ]; then
  NOTARY_MODE="profile"
elif [ -n "${LIVEFT_NOTARY_KEY_FILE:-}" ] && [ -n "${LIVEFT_NOTARY_KEY_ID:-}" ] && [ -n "${LIVEFT_NOTARY_ISSUER:-}" ]; then
  NOTARY_MODE="api_key"
elif [ -n "${LIVEFT_NOTARY_APPLE_ID:-}" ] && [ -n "${LIVEFT_NOTARY_TEAM_ID:-}" ] && [ -n "${LIVEFT_NOTARY_PASSWORD:-}" ]; then
  NOTARY_MODE="apple_id"
else
  printf 'Set LIVEFT_NOTARY_PROFILE, API key credentials, or Apple ID credentials before notarizing.\n' >&2
  exit 1
fi

TARGET_PATH=$(resolve_path "${TARGET_INPUT}")
if [ ! -e "${TARGET_PATH}" ]; then
  printf 'Notarization target not found: %s\n' "${TARGET_PATH}" >&2
  exit 1
fi

case "${TARGET_PATH}" in
  *.app|*.dmg)
    NOTARIZED_ANY=1
    notarize_target "${TARGET_PATH}"
    ;;
  *)
    if [ ! -d "${TARGET_PATH}" ]; then
      printf 'Unsupported notarization target: %s\n' "${TARGET_PATH}" >&2
      exit 1
    fi

    for candidate in "${TARGET_PATH}"/LiveFT*.dmg; do
      [ -e "${candidate}" ] || continue
      NOTARIZED_ANY=1
      notarize_target "${candidate}"
    done

    if [ "${NOTARIZED_ANY}" -ne 1 ]; then
      for candidate in "${TARGET_PATH}"/LiveFT*.app; do
        [ -e "${candidate}" ] || continue
        NOTARIZED_ANY=1
        notarize_target "${candidate}"
      done
    fi
    ;;
esac

if [ "${NOTARIZED_ANY}" -ne 1 ]; then
  printf 'No LiveFT app bundles or DMGs found under %s\n' "${TARGET_PATH}" >&2
  exit 1
fi
