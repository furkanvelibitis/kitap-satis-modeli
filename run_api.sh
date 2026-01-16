#!/bin/bash
# =============================================================
# API BAŞLATMA SCRIPTİ
# =============================================================

echo "=============================================="
echo "KİTAP SATIŞ TAHMİN API BAŞLATILIYOR"
echo "=============================================="

# Gerekli paketleri kur
pip install fastapi uvicorn python-multipart pillow

# Proje klasörüne git
cd /workspace/proje/src

# GPU kontrolü
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# API'yi başlat
echo ""
echo "API başlatılıyor: http://0.0.0.0:8000"
echo "Docs: http://0.0.0.0:8000/docs"
echo ""

uvicorn api:app --host 0.0.0.0 --port 8000

