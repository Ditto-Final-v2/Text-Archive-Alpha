@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found at .venv\Scripts\python.exe
  echo Run run_textarchive.bat first.
  exit /b 1
)

set PASS=%~1
if "%PASS%"=="" (
  set /p PASS=Enter app password for smoke test: 
)

".venv\Scripts\python.exe" smoke_test.py --password "%PASS%"
set CODE=%ERRORLEVEL%
echo.
if %CODE% NEQ 0 (
  echo Smoke tests failed.
) else (
  echo Smoke tests passed.
)
pause
exit /b %CODE%
