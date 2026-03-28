#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
DIST_DIR="${1:-dist}"
KEEP_APP_AFTER_DMG="${LIVEFT_KEEP_APP_AFTER_DMG:-0}"

case "${DIST_DIR}" in
  /*) DIST_PATH="${DIST_DIR}" ;;
  *) DIST_PATH="${REPO_ROOT}/${DIST_DIR}" ;;
esac

cd "${DIST_PATH}"
app_dir="$(find . -maxdepth 1 -type d -name 'LiveFT*.app' | head -n 1)"

if [ -z "${app_dir}" ]; then
  echo "No app bundle found in ${DIST_PATH}" >&2
  exit 1
fi

app_dir="${app_dir#./}"
app_name="${app_dir%.app}"

hdiutil create -ov -format UDZO -fs HFS+ -srcfolder "${app_dir}" -volname "${app_name}" "${app_name}.dmg"

if [ -d "${app_name}" ]; then
  rm -rf "${app_name}"
fi

if [ "${KEEP_APP_AFTER_DMG}" != "1" ]; then
  rm -rf "${app_dir}"
fi
