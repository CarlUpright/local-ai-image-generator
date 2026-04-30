@echo off
echo ============================================================
echo   OpenDalle V1.1 -- Generateur d'images local
echo ============================================================
echo.

call venv\Scripts\activate

rem Charger le token HuggingFace depuis .env si disponible
if exist .env (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if "%%a"=="HF_TOKEN" set HF_TOKEN=%%b
    )
)

if defined HF_TOKEN (
    echo Token HuggingFace detecte : telechargement rapide active.
) else (
    echo Aucun token HuggingFace. Creez .env avec HF_TOKEN=hf_xxx pour 50+ MB/s.
)
echo.

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
