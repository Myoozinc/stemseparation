#!/usr/bin/env bash
# ==============================================================================
# Myooz Labs - Iniciar Motor Local de Generación Musical (Apple Silicon Mac)
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "=================================================================="
echo " 🎵 MYOOZ LABS - INICIADOR DE MOTOR LOCAL (APPLE SILICON)"
echo "=================================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 no está instalado en el sistema."
    exit 1
fi

# Check or create virtual environment for isolation
VENV_DIR=".venv_local"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creando entorno virtual local aislado ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Check dependencies
echo "[*] Verificando librerías requeridas (torch, transformers, scipy, soundfile)..."
python3 -c "import torch, transformers, scipy, soundfile" 2>/dev/null || {
    echo "[*] Instalando dependencias de IA para Apple Silicon..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio transformers scipy soundfile
}

echo "[✓] Entorno verificado con éxito."
echo "[*] Iniciando servidor local en http://localhost:7860 ..."
echo "------------------------------------------------------------------"
echo "Abre tu navegador en https://stemseparation.vercel.app/generator.html"
echo "(o http://localhost:3000/generator.html) y verás el motor LOCAL activo."
echo "Presiona Ctrl+C para detener el motor."
echo "------------------------------------------------------------------"

python3 local_generator.py
