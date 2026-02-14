# Text Archive Alpha (Local App)

This build runs entirely on your own computer.

## Quick Start (Windows)
1. Double-click `Launch_this_for_Windows.bat`
2. Wait for dependency install (first run can take a few minutes)
3. Your browser opens to `http://127.0.0.1:8000`
4. First run goes to setup:
   - Upload your XML or CSV export
   - Enter your name and your partner's name
   - Set your app password
   - Click **Build My Archive**

Windows launcher preflight checks:
- Python 3.11.x installed (recommended: 3.11.6)
- Microsoft Visual C++ Redistributable (x64) installed

Python install note (Windows):
- During Python setup, check the box: `Add Python to PATH`

## Quick Start (macOS)
macOS notes:
- Works on both Apple Silicon (`arm64`) and Intel (`x86_64`) Macs.
- Required Python version: `3.11.x` (recommended: `3.11.6`).
- Install Python (macOS):
  - Python macOS downloads: `https://www.python.org/downloads/macos/`
- Xcode Command Line Tools are required (`xcode-select --install`).
- Install Xcode Command Line Tools / Xcode:
  - Apple Developer downloads (Command Line Tools): `https://developer.apple.com/download/all/?q=command%20line%20tools`
  - Xcode (App Store): `https://apps.apple.com/app/xcode/id497799835`

1. Open Terminal in this folder
2. Run:
```bash
chmod +x run_textarchive.command
./run_textarchive.command
```
3. Launcher starts the server in the background, waits for readiness, then opens `http://127.0.0.1:8000`

### Optional: one-time `.app` wrapper (double-click launch)
If you want a clickable Mac app icon:

```bash
chmod +x create_mac_app.command
./create_mac_app.command
```

This creates `Launch_this_for_Mac.app` in the same folder.
After that, users can launch by double-clicking the app.

## What the launcher does
- Creates a local `.venv` virtual environment
- Installs packages from `requirements.txt`
- Starts the FastAPI app with Uvicorn
- Opens the local URL in your browser
- During setup, the app now builds both:
  - the message DB + FAISS index
  - signal analysis scores for Signals Timeline
- Setup uses a faster default embedding model (`sentence-transformers/all-MiniLM-L6-v2`) for better CPU performance during alpha onboarding.
- Setup auto-detects CUDA and uses GPU for DB/signal builds when available; otherwise it falls back to CPU.

## Setup status + reset
- After build, setup now shows a summary page:
  - success with next steps
  - failure with readable error details, recent log tail, and a copy button
- A **Reset Archive** button is available on setup/login pages to clear local archive data and restart setup.
- Reset Archive now clears all local archive artifacts:
  - `texts.db`, `texts.faiss`, `setup_state.json`
  - files under `uploads/` and `attachments/` (including previews)
  - local `__pycache__/` folders
  - runtime logs (keeps `logs/app.log` file but truncates it)
- On setup failures, the page now shows the exact log file path so non-technical users can share diagnostics.

## Logs
- Runtime events are written to:
  - `logs/app.log`
- Includes setup/build start+failure+success and auth events.

## Smoke test (release sanity check)
Run this after setup to quickly verify core routes/features:

Windows:
```powershell
.\run_smoke_tests.bat YOUR_PASSWORD
```

or:
```powershell
.\.venv\Scripts\python.exe smoke_test.py --password "YOUR_PASSWORD"
```

macOS/Linux:
```bash
source .venv/bin/activate
python smoke_test.py --password "YOUR_PASSWORD"
```

Optional flags:
- `--base-url http://127.0.0.1:8000`
- `--timeout 20`

## Stopping the app
- Go to the terminal window and press `Ctrl+C`.
- Or from the homepage in the browser, click **Stop Program** (under **Logout**).

macOS note:
- The launcher runs the server in background mode (`nohup`), so there may not be an active terminal session to `Ctrl+C`.
- Use the in-app **Stop Program** button to shut it down cleanly.

## Troubleshooting
- `Python was not found`:
  - Install Python 3.10+ and try again.
- Port already in use:
  - Close other app using `8000`, or run manually on another port.
- Slow first run:
  - Model downloads can take time on first launch; later runs are faster.
- Setup/build error:
  - Verify your file is a valid XML/CSV export, then try again.

## Manual run (optional)
```bash
python -m venv .venv
```
Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Then:
```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
