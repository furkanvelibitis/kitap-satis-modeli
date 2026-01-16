"""
Kitap Satış Tahmin Modeli - Veri İndirme Scripti
"""

import os
from datasets import load_dataset
import pandas as pd

# Proje dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def download_kitapyurdu_reviews():
    """HuggingFace'den Kitapyurdu yorumları veri setini indir"""
    print("=" * 50)
    print("HuggingFace - Kitapyurdu Yorumları indiriliyor...")
    print("=" * 50)

    try:
        # Veri setini yükle
        dataset = load_dataset("alibayram/kitapyurdu_yorumlar")

        # Train split'i DataFrame'e çevir
        df = dataset['train'].to_pandas()

        # CSV olarak kaydet
        output_path = os.path.join(DATA_DIR, "kitapyurdu_yorumlar.csv")
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\n✓ Başarıyla indirildi!")
        print(f"  Toplam yorum sayısı: {len(df):,}")
        print(f"  Sütunlar: {list(df.columns)}")
        print(f"  Dosya: {output_path}")

        return df

    except Exception as e:
        print(f"✗ Hata: {e}")
        return None

def download_kaggle_books():
    """Kaggle'dan Turkish Book veri setini indir"""
    print("\n" + "=" * 50)
    print("Kaggle - Turkish Book Dataset bilgisi")
    print("=" * 50)

    print("""
Kaggle veri seti için manuel indirme gerekli:

1. https://www.kaggle.com/datasets adresine git
2. "Turkish Book Data Set" ara (Muhammed İbrahim Top)
3. Veri setini indir ve şu klasöre koy:
   {data_dir}

VEYA Kaggle API kullanmak için:
   pip install kaggle
   kaggle datasets download -d <dataset-name>
""".format(data_dir=DATA_DIR))

    # Kaggle veri seti zaten varsa kontrol et
    possible_files = ["turkish_books.csv", "books.csv", "kitaplar.csv"]
    for fname in possible_files:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"✓ Kaggle verisi bulundu: {fpath}")
            return pd.read_csv(fpath)

    return None

if __name__ == "__main__":
    # Dizinlerin var olduğundan emin ol
    os.makedirs(DATA_DIR, exist_ok=True)

    # HuggingFace veri setini indir
    reviews_df = download_kitapyurdu_reviews()

    # Kaggle veri seti hakkında bilgi ver
    books_df = download_kaggle_books()

    print("\n" + "=" * 50)
    print("İndirme işlemi tamamlandı!")
    print("=" * 50)
