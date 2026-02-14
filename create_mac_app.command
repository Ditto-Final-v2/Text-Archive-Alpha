#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v osacompile >/dev/null 2>&1; then
  echo "osacompile was not found."
  echo "This script must be run on macOS."
  read -r -p "Press Enter to close..."
  exit 1
fi

LAUNCHER_PATH="$(pwd)/run_textarchive.command"
APP_NAME="Launch_this_for_Mac.app"
APP_PATH="$(pwd)/${APP_NAME}"

if [ ! -f "${LAUNCHER_PATH}" ]; then
  echo "Missing launcher script: ${LAUNCHER_PATH}"
  read -r -p "Press Enter to close..."
  exit 1
fi

chmod +x "${LAUNCHER_PATH}"

APPLE_SCRIPT_FILE="$(mktemp /tmp/textarchive_applescript.XXXXXX)"
cat > "${APPLE_SCRIPT_FILE}" <<EOF
do shell script "cd " & quoted form of "$(pwd)" & " && chmod +x run_textarchive.command && ./run_textarchive.command >/tmp/textarchive_launcher.log 2>&1 &"
EOF

rm -rf "${APP_PATH}"
osacompile -o "${APP_PATH}" "${APPLE_SCRIPT_FILE}"
rm -f "${APPLE_SCRIPT_FILE}"

echo "Created: ${APP_PATH}"
echo "You can now double-click '${APP_NAME}' to launch Text Archive."
read -r -p "Press Enter to close..."
