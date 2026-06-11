@echo off
chcp 65001 >nul
title Econometrie financiere - Indices boursiers
cd /d "%~dp0"

echo ============================================================
echo   Application d'econometrie financiere
echo   ADF - ARCH-LM - GARCH - Granger - Forbes-Rigobon
echo ============================================================
echo.

REM Verifie que Streamlit est installe
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [INFO] Installation des dependances en cours...
    python -m pip install -r requirements.txt
    echo.
)

echo [INFO] Lancement de l'application dans votre navigateur...
echo [INFO] Pour arreter : fermez cette fenetre ou faites Ctrl+C.
echo.
python -m streamlit run app.py

pause
