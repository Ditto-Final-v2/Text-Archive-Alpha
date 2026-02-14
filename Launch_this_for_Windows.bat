@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8000"
set "PY_CMD="

echo Running preflight checks...

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -V >nul 2>nul
  if %errorlevel%==0 (
    set "PY_CMD=py -3.11"
  ) else (
    set "PY_CMD=py -3"
  )
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY_CMD=python"
  ) else (
    echo.
    echo [ERROR] Python 3.11.x was not found.
    echo Install Python first:
    echo https://www.python.org/downloads/windows/
    echo During install, check: "Add Python to PATH"
    echo.
    pause
    exit /b 1
  )
)

%PY_CMD% -c "import sys; raise SystemExit(0 if (sys.version_info.major==3 and sys.version_info.minor==11) else 1)"
if errorlevel 1 (
  echo.
  echo [ERROR] Python 3.11.x is required for this alpha build.
  echo Install Python 3.11.6:
  echo https://www.python.org/downloads/windows/
  echo During install, check: "Add Python to PATH"
  echo.
  pause
  exit /b 1
)

if not exist "%SystemRoot%\System32\vcruntime140.dll" (
  echo.
  echo [ERROR] Microsoft Visual C++ Redistributable is missing.
  echo Install the x64 package from:
  echo https://aka.ms/vs/17/release/vc_redist.x64.exe
  echo.
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" goto have_venv

%PY_CMD% -m venv .venv

:have_venv
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Could not activate virtual environment.
  pause
  exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
  echo pip upgrade failed.
  pause
  exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed.
  pause
  exit /b 1
)

rem If app is already running, just open browser and exit.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://%HOST%:%PORT%/' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if %errorlevel%==0 (
  start "" "http://%HOST%:%PORT%"
  endlocal
  exit /b 0
)

rem If port is occupied by something else, don't launch a second server.
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
  echo [ERROR] Port %PORT% is already in use by another process.
  echo Close the process using %PORT%, or change the port in run_textarchive.bat.
  echo If TextArchive is already running, open: http://%HOST%:%PORT%
  pause
  endlocal
  exit /b 1
)

start "TextArchive Server" "%ComSpec%" /k "cd /d ""%~dp0"" && call "".venv\Scripts\activate.bat"" && python -m uvicorn app:app --host %HOST% --port %PORT%"

powershell -NoProfile -Command "$u='http://%HOST%:%PORT%/'; $ok=$false; for($i=0; $i -lt 90; $i++){ try { Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 1 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ok){ Start-Process $u; exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 (
  echo [WARN] Server did not respond in time. Open http://%HOST%:%PORT% manually.
)

endlocal
exit /b 0
