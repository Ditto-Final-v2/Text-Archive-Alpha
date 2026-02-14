#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="127.0.0.1"
PORT="8000"
BASE_URL="http://${HOST}:${PORT}"
ARCH="$(uname -m)"
PY_CMD=""
PID_FILE=".server.pid"
LOG_FILE="logs/server.log"

echo "Running preflight checks..."
echo "Detected architecture: ${ARCH}"

if command -v python3.11 >/dev/null 2>&1; then
  PY_CMD="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
else
  echo "Python was not found."
  echo "Install Python 3.11.6 and try again:"
  echo "https://www.python.org/downloads/macos/"
  read -r -p "Press Enter to close..."
  exit 1
fi

if ! "${PY_CMD}" -c "import sys; raise SystemExit(0 if (sys.version_info.major==3 and sys.version_info.minor==11) else 1)"; then
  echo "Python 3.11.x is required for this alpha build."
  echo "Install Python 3.11.6:"
  echo "https://www.python.org/downloads/macos/"
  read -r -p "Press Enter to close..."
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are required."
  echo "Run this command, then run launcher again:"
  echo "xcode-select --install"
  echo "If the command fails, use Apple Developer downloads:"
  echo "https://developer.apple.com/download/all/?q=command%20line%20tools"
  echo "You can also install full Xcode from the App Store:"
  echo "https://apps.apple.com/app/xcode/id497799835"
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  "${PY_CMD}" -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install --upgrade pip
if ! pip install -r requirements.txt; then
  echo
  echo "Dependency install failed."
  echo "Architecture: ${ARCH}"
  echo "If this is an Intel Mac, ensure macOS is reasonably up-to-date and retry."
  echo "If this is Apple Silicon, ensure Python is arm64-native (not Rosetta)."
  read -r -p "Press Enter to close..."
  exit 1
fi

mkdir -p logs

# If already running, just open browser and exit.
if curl -fsS --max-time 1 "${BASE_URL}/" >/dev/null 2>&1; then
  open "${BASE_URL}" >/dev/null 2>&1 || true
  exit 0
fi

# If something else is using the port, fail clearly.
if lsof -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[ERROR] Port ${PORT} is already in use by another process."
  echo "Close the process using ${PORT}, or change the port in run_textarchive.command."
  echo "If TextArchive is already running, open: ${BASE_URL}"
  read -r -p "Press Enter to close..."
  exit 1
fi

# Start server detached so launcher can exit.
nohup python -m uvicorn app:app --host "${HOST}" --port "${PORT}" >> "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

# Wait for server readiness, then open browser.
READY=0
for _i in {1..90}; do
  if curl -fsS --max-time 1 "${BASE_URL}/" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" -eq 1 ]; then
  open "${BASE_URL}" >/dev/null 2>&1 || true
  exit 0
fi

echo "[WARN] Server did not respond in time. Open ${BASE_URL} manually."
echo "Check logs at ${LOG_FILE}."
read -r -p "Press Enter to close..."
exit 1
