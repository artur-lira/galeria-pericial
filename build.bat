@echo off
setlocal
REM =====================================================================
REM  Gera o executavel Windows: dist\GaleriaPericial\GaleriaPericial.exe
REM  Uso:  build.bat         -> versao com GPU NVIDIA CUDA (1 ou varias GPUs)
REM        build.bat cpu     -> versao somente CPU (menor)
REM  Requisito: Python 3.11-3.13 64 bits (python.org) com "Add to PATH".
REM  Para CUDA basta o driver NVIDIA atualizado (o CUDA vem junto do PyTorch).
REM =====================================================================
cd /d "%~dp0"
set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if /I "%1"=="cpu" set TORCH_INDEX=https://download.pytorch.org/whl/cpu
if not exist .venv ( python -m venv .venv || goto :erro )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :erro
pip install torch torchvision --index-url %TORCH_INDEX% || goto :erro
pip install pyinstaller || goto :erro

echo === Baixando os pesos da CNN (ResNet-50 ImageNet, ~100 MB) ===
if not exist modelos mkdir modelos
if not exist modelos\resnet50-0676ba61.pth (
  python -c "import torch; torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet50-0676ba61.pth', r'modelos\resnet50-0676ba61.pth', hash_prefix='0676ba61')" || goto :erro
)

echo === Autoteste do motor ===
python -c "from galeria import engine as e; x=e.Engine(); print(x.describe()); import sys; sys.exit(0 if x.ok else 1)" || goto :erro

pyinstaller --noconfirm --clean --windowed --name GaleriaPericial ^
  --collect-all pillow_heif --collect-all rawpy --hidden-import torchvision ^
  main.py || goto :erro
xcopy /E /I /Y modelos dist\GaleriaPericial\modelos >nul
echo.
echo ==== OK: dist\GaleriaPericial\GaleriaPericial.exe ====
echo Copie a pasta dist\GaleriaPericial inteira (inclui a pasta "modelos").
pause
exit /b 0
:erro
echo.
echo **** Falha no build. Veja as mensagens acima. ****
pause
exit /b 1
