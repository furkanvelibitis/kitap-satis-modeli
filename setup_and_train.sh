#!/bin/bash
# =============================================================
# KITAP SATIS TAHMIN MODELI - BULUT EGITIM SCRIPTI
# Vast.ai / RunPod icin hazirlanmistir
# =============================================================

echo "=============================================="
echo "KURULUM BASLIYOR..."
echo "=============================================="

# 1. Sistem guncelle ve gerekli paketleri kur
apt-get update
apt-get install -y unzip wget

# 2. Python paketlerini kur
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn pillow tqdm

# 3. Zip dosyasini cikart (eger cikarilmamissa)
if [ -f "kitap_project.zip" ]; then
    echo "Zip dosyasi cikartiliyor..."
    unzip -o kitap_project.zip
fi

# 4. Klasor yapisini kontrol et
echo ""
echo "Klasor yapisi:"
ls -la

# 5. Data klasorunu kontrol et
echo ""
echo "Data klasoru:"
ls -la data/

echo ""
echo "Covers klasoru (ilk 10):"
ls data/covers/ | head -10
echo "Toplam kapak sayisi: $(ls data/covers/*.jpg 2>/dev/null | wc -l)"

# 6. GPU kontrolu
echo ""
echo "=============================================="
echo "GPU KONTROLU"
echo "=============================================="
python -c "import torch; print(f'CUDA mevcut: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"YOK\"}')"

# 7. Veri hazirlama (eger processed klasoru bossa)
if [ ! -f "data/processed/train.csv" ]; then
    echo ""
    echo "=============================================="
    echo "VERI HAZIRLANIYOR..."
    echo "=============================================="
    cd src
    python data_preparation.py
    cd ..
fi

# 8. Egitimi baslat
echo ""
echo "=============================================="
echo "EGITIM BASLIYOR!"
echo "=============================================="
cd src
python train.py

# 9. Sonuclari goster
echo ""
echo "=============================================="
echo "EGITIM TAMAMLANDI!"
echo "=============================================="
echo ""
echo "Model dosyalari:"
ls -la ../models/

echo ""
echo "Simdi 'models/' klasorunu indirebilirsin."
echo "=============================================="
