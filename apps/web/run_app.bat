@echo off
rem Launch the web app (app.py) on Windows -- the counterpart of run_app.sh.
rem Double-click it, or run it from a terminal to pass options: run_app.bat --port 8080
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe app.py %*
) else (
    python app.py %*
)
rem Keep the window open after an error, so a double-click shows why.
if errorlevel 1 pause
