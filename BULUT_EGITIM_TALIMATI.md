# BULUT GPU EĞİTİM TALİMATI

## ADIM 1: Projeyi Zipke

Windows'ta şu klasörleri zipke:
```
kitap satış modeli/
├── src/           (tüm python dosyaları)
├── data/
│   ├── raw/       (books.csv)
│   └── covers/    (tüm jpg dosyaları)
├── models/        (boş olabilir)
└── setup_and_train.sh
```

**Zip adı:** `kitap_project.zip`

---

## ADIM 2: Vast.ai Hesabı Aç

1. https://vast.ai adresine git
2. Hesap oluştur (Google ile giriş yapabilirsin)
3. Sol menüden "Billing" → $5-10 yükle (Kredi kartı veya kripto)

---

## ADIM 3: GPU Kirala

1. https://vast.ai/console/create/ adresine git
2. Filtreler:
   - **GPU:** RTX 3090 veya RTX 4090 (hızlı)
   - **Disk:** 50 GB (covers için yeterli)
   - **Image:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`

3. En ucuz olan "RENT" butonuna tıkla (~$0.20-0.40/saat)

---

## ADIM 4: Makineye Bağlan

1. "Instances" sayfasında makineni gör
2. "Open" butonuna tıkla → Terminal açılır
3. Veya SSH ile bağlan (SSH komutunu kopyala)

---

## ADIM 5: Dosyaları Yükle

**Yöntem A - Web arayüzü:**
- Vast.ai arayüzünde "Jupyter" aç
- Upload butonu ile `kitap_project.zip` yükle

**Yöntem B - SCP ile (daha hızlı):**
```bash
scp -P PORT kitap_project.zip root@IP:/workspace/
```
(PORT ve IP'yi Vast.ai'dan al)

---

## ADIM 6: Eğitimi Başlat

Terminalde şu komutları çalıştır:

```bash
cd /workspace

# Zip'i çıkart
unzip kitap_project.zip

# Klasöre gir
cd "kitap satis modeli"

# Eğitimi başlat
bash setup_and_train.sh
```

---

## ADIM 7: Bekle

Eğitim süresi (RTX 3090 ile):
- Veri hazırlama: ~5 dakika
- Aşama 1 (Baseline): ~10 dakika
- Aşama 2 (Kapak): ~30-45 dakika
- Aşama 3 (Fine-tune): ~20 dakika

**Toplam: ~1-1.5 saat**

---

## ADIM 8: Modeli İndir

Eğitim bitince:

```bash
# Models klasörünü zipke
zip -r trained_models.zip models/
```

Sonra Jupyter arayüzünden `trained_models.zip` indir.

---

## ADIM 9: Makineyi Kapat

**ÖNEMLİ:** İşin bitince "DESTROY" butonuna bas!
Yoksa para akmaya devam eder.

---

## SORUN GİDERME

**"CUDA out of memory" hatası:**
- `src/train.py` içinde `batch_size=16` → `batch_size=8` yap

**"No such file" hatası:**
- Klasör yapısını kontrol et, zip doğru çıkmış mı?

**İnternet yavaş, yükleme uzun sürüyor:**
- Covers klasörünü ayrı zipke (~1.5 GB)
- Google Drive'a yükle, makineden `wget` ile çek

---

## TOPLAM MALİYET

- GPU: ~$0.30/saat × 1.5 saat = ~$0.50
- Ekstra güvenlik payı: ~$1

**Toplam: $1-2 (₺35-70)**
