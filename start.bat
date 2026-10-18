@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First follow the setup instructions in README.md to create the .venv environment.
  pause
  exit /b 1
)
if not exist "artifacts\model.joblib" (
  echo Train the model first: .venv\Scripts\python.exe train.py
  pause
  exit /b 1
)
echo Flowcast - Traffic Volume Prediction
echo Open http://127.0.0.1:5000 in your browser after the server starts.
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
