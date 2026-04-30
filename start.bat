@echo off
echo ============================================================
echo   OpenDalle V1.1 -- Generateur d'images local
echo ============================================================
echo.

call venv\Scripts\activate

echo [1/2] Verification / telechargement du modele...
python download_model.py
if %errorlevel% neq 0 (
    echo ERREUR lors du telechargement.
    pause
    exit /b 1
)

echo.
echo [2/2] Lancement de l'interface web...
python app.py
pause
