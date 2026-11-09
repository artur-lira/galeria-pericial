@echo off
REM Executa direto do codigo-fonte (rode build.bat uma vez antes, para criar o ambiente e baixar o modelo)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python main.py %*
