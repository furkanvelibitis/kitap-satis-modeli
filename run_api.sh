#!/bin/bash
# =============================================================
# API BAŞLATMA SCRIPTİ (Satış Tahmini + LoRA Kapak Üretimi)
# =============================================================

echo "=============================================="
echo "KİTAP SATIŞ TAHMİN API + LORA BAŞLATILIYOR"
echo "=============================================="

# Gerekli paketleri kur
echo "Bağımlılıklar yükleniyor..."
pip install fastapi uvicorn python-multipart pillow

# LoRA için ek paketler (diffusers, peft, vb.)
pip install diffusers transformers accelerate peft safetensors

# Proje klasörüne git
cd /workspace/proje/src

# GPU kontrolü
echo ""
echo "GPU Durumu:"
python -c "
import torch
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
"

# LoRA model kontrolü
echo ""
echo "Model Durumu:"
if [ -d "../models/lora_sdxl/final" ]; then
    echo "  LoRA modeli: MEVCUT"
else
    echo "  LoRA modeli: YOK (kapak üretimi çalışmayacak)"
fi

if [ -f "../models/two_stage_model.pth" ]; then
    echo "  Satış tahmin modeli: MEVCUT"
elif [ -f "../models/baseline_model.pth" ]; then
    echo "  Satış tahmin modeli: SADECE BASELINE"
else
    echo "  Satış tahmin modeli: YOK"
fi

# API'yi başlat
echo ""
echo "=============================================="
echo "API başlatılıyor: http://0.0.0.0:8000"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  - POST /predict          : Satış tahmini"
echo "  - POST /generate-cover   : LoRA ile kapak üretimi"
echo "  - GET  /lora-status      : LoRA durumu"
echo "  - GET  /health           : Sistem durumu"
echo "  - GET  /docs             : Swagger UI"
echo ""

uvicorn api:app --host 0.0.0.0 --port 8000

