#!/bin/bash
# =============================================================
# LORA EĞİTİM SCRIPTI - KİTAP KAPAĞI ÜRETİCİ
# =============================================================

echo "=============================================="
echo "LORA EĞİTİMİ BAŞLIYOR"
echo "=============================================="

# Gerekli paketleri kur
pip install diffusers transformers accelerate peft bitsandbytes xformers safetensors

# GPU kontrolü
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"YOK\"}')"

# Proje klasörüne git
cd /workspace/proje/src

# 1. Önce data_preparation çalıştır (good_covers_for_lora.csv oluşsun)
if [ ! -f "../data/processed/good_covers_for_lora.csv" ]; then
    echo "Veri hazırlığı yapılıyor..."
    python data_preparation.py
fi

# 2. LoRA dataset hazırla ve eğit
echo ""
echo "LoRA eğitimi başlıyor..."
python train_lora.py --max_images 300 --max_steps 1500 --lora_rank 32

echo ""
echo "=============================================="
echo "LORA EĞİTİMİ TAMAMLANDI!"
echo "=============================================="
echo "Model: ../models/lora_sdxl/final"
