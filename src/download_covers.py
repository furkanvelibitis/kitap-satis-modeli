"""
Kitap Kapak Resimlerini İndirme Scripti
10+ yorum alan kitapların kapaklarını indirir (30,978 kitap)
Tahmini süre: 15-20 dakika (20 paralel ile)
"""

import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Ayarlar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
COVERS_DIR = os.path.join(BASE_DIR, "data", "covers")
PARALLEL_DOWNLOADS = 20  # Paralel indirme sayısı
MIN_REVIEWS = 10  # Minimum yorum sayısı filtresi

def get_300px_url(url):
    """URL'yi 300px genişliğe çevir"""
    if pd.isna(url):
        return None
    return url.replace('wi:100', 'wi:300')

def download_image(args):
    """Tek bir resmi indir"""
    idx, url, save_path = args

    if os.path.exists(save_path):
        return idx, True, "Zaten var"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return idx, True, "OK"
        else:
            return idx, False, f"HTTP {response.status_code}"
    except Exception as e:
        return idx, False, str(e)[:50]

def main():
    # Klasörü oluştur
    os.makedirs(COVERS_DIR, exist_ok=True)

    # Veriyi yükle
    print("Veri yükleniyor...")
    df = pd.read_csv(os.path.join(DATA_DIR, "books.csv"), low_memory=False)

    # Filtrele: 10+ yorum
    df_filtered = df[df['reviews'] >= MIN_REVIEWS].copy()
    print(f"Toplam kitap: {len(df):,}")
    print(f"10+ yorumlu kitap: {len(df_filtered):,}")

    # İndirme listesi hazırla
    download_tasks = []
    for idx, row in df_filtered.iterrows():
        url = get_300px_url(row['image'])
        if url:
            filename = f"{row['isbn']}.jpg"
            save_path = os.path.join(COVERS_DIR, filename)
            download_tasks.append((idx, url, save_path))

    print(f"İndirilecek resim: {len(download_tasks):,}")
    print(f"Paralel indirme: {PARALLEL_DOWNLOADS}")
    print()

    # Zaten indirilmiş olanları say
    existing = sum(1 for _, _, path in download_tasks if os.path.exists(path))
    if existing > 0:
        print(f"Zaten indirilmiş: {existing:,}")

    print("=" * 50)
    print("İndirme başlıyor...")
    print("=" * 50)

    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0

    # Paralel indirme
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as executor:
        futures = {executor.submit(download_image, task): task for task in download_tasks}

        with tqdm(total=len(download_tasks), desc="İndiriliyor", unit="resim") as pbar:
            for future in as_completed(futures):
                idx, ok, msg = future.result()
                if msg == "Zaten var":
                    skipped += 1
                elif ok:
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    elapsed = time.time() - start_time

    print()
    print("=" * 50)
    print("TAMAMLANDI!")
    print("=" * 50)
    print(f"Süre: {elapsed/60:.1f} dakika")
    print(f"Başarılı: {success:,}")
    print(f"Zaten vardı: {skipped:,}")
    print(f"Başarısız: {failed:,}")
    print(f"Klasör: {COVERS_DIR}")

if __name__ == "__main__":
    main()
