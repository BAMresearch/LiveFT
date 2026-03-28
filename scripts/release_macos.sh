#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VERSION_SUFFIX="${1:-}"
OS_TAG="${2:-}"

has_notary_credentials() {
  [ -n "${LIVEFT_NOTARY_PROFILE:-}" ] ||
    { [ -n "${LIVEFT_NOTARY_KEY_FILE:-}" ] && [ -n "${LIVEFT_NOTARY_KEY_ID:-}" ] && [ -n "${LIVEFT_NOTARY_ISSUER:-}" ]; } ||
    { [ -n "${LIVEFT_NOTARY_APPLE_ID:-}" ] && [ -n "${LIVEFT_NOTARY_TEAM_ID:-}" ] && [ -n "${LIVEFT_NOTARY_PASSWORD:-}" ]; }
}

if [ "$(uname -s)" != "Darwin" ]; then
  printf 'scripts/release_macos.sh must be run on macOS.\n' >&2
  exit 1
fi

export LIVEFT_KEEP_APP_AFTER_DMG="${LIVEFT_KEEP_APP_AFTER_DMG:-1}"

sh "${SCRIPT_DIR}/build_dist.sh" "${VERSION_SUFFIX}" "${OS_TAG}"

if [ -n "${LIVEFT_CODESIGN_IDENTITY:-}" ]; then
  sh "${SCRIPT_DIR}/sign_macos.sh" dist
fi

sh "${SCRIPT_DIR}/create_dmg.sh" dist

if [ -n "${LIVEFT_CODESIGN_IDENTITY:-}" ]; then
  for candidate in "${REPO_ROOT}"/dist/LiveFT*.dmg; do
    [ -e "${candidate}" ] || continue
    sh "${SCRIPT_DIR}/sign_macos.sh" "${candidate}"
  done
fi

if has_notary_credentials; then
  sh "${SCRIPT_DIR}/notarize_macos.sh" dist
  export LIVEFT_EXPECT_NOTARIZED=1
fi

sh "${SCRIPT_DIR}/verify_macos.sh" dist
