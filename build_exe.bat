@echo off
setlocal
title Hospital Treatment Cost Predictor - EXE Builder

echo.
echo ================================================
echo   Hospital Treatment Cost Predictor
echo   Windows EXE Builder
echo ================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python 3.12 from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist .venv (
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/4] Building EXE...
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "HospitalTreatmentCostPredictor" ^
  --add-data "model_file.joblib;." ^
  --collect-all sklearn ^
  --collect-all joblib ^
  app.py

echo.
echo [4/4] Finished.
echo.
echo Your EXE is here:
echo dist\HospitalTreatmentCostPredictor.exe
echo.
pause
