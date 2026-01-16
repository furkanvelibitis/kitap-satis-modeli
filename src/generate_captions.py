"""
Kapak Resimleri için Otomatik Caption Üretici
BLIP modeli kullanarak kitap kapaklarına açıklama ekler

Kullanım:
    python generate_captions.py --input_dir ../data/lora_dataset/train
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        return True
    except ImportError:
        print("Gerekli paketler yükleniyor...")
        os.system("pip install transformers pillow torch")
        return False


def generate_captions_blip(input_dir, prefix="book cover, bestseller, "):
    """
    BLIP modeli ile caption üret

    Args:
        input_dir: Resim klasörü
        prefix: Her caption'ın başına eklenecek metin
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("BLIP modeli yükleniyor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # Resim dosyalarını bul
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"{len(image_files)} resim bulundu")

    for img_file in tqdm(image_files, desc="Caption üretiliyor"):
        img_path = os.path.join(input_dir, img_file)
        txt_path = os.path.join(input_dir, img_file.rsplit('.', 1)[0] + '.txt')

        try:
            # Resmi yükle
            image = Image.open(img_path).convert('RGB')

            # Caption üret
            inputs = processor(image, return_tensors="pt").to(device, torch.float16 if torch.cuda.is_available() else torch.float32)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            caption = processor.decode(out[0], skip_special_tokens=True)

            # Prefix ekle ve kaydet
            full_caption = f"{prefix}{caption}, high quality, professional design"

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_caption)

        except Exception as e:
            print(f"Hata ({img_file}): {e}")
            continue

    print(f"\nCaption'lar oluşturuldu: {input_dir}")


def generate_simple_captions(input_dir, csv_path=None):
    """
    Basit caption üretici - CSV'den kitap bilgilerini kullanır

    Args:
        input_dir: Resim klasörü
        csv_path: good_covers_for_lora.csv yolu
    """
    import pandas as pd

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        isbn_to_info = {str(row['isbn']): row for _, row in df.iterrows()}
    else:
        isbn_to_info = {}

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in tqdm(image_files, desc="Caption yazılıyor"):
        isbn = img_file.rsplit('.', 1)[0]
        txt_path = os.path.join(input_dir, isbn + '.txt')

        # Kitap bilgisi varsa kullan
        if isbn in isbn_to_info:
            info = isbn_to_info[isbn]
            author = info.get('author', 'unknown author')
            caption = f"book cover, bestseller, Turkish book, by {author}, professional design, high quality"
        else:
            caption = "book cover, bestseller, Turkish book, professional design, high quality, appealing"

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)

    print(f"\n{len(image_files)} caption oluşturuldu")


def main():
    parser = argparse.ArgumentParser(description='Kapak Resimlerine Caption Ekle')
    parser.add_argument('--input_dir', type=str, required=True, help='Resim klasörü')
    parser.add_argument('--mode', choices=['blip', 'simple'], default='simple', help='Caption modu')
    parser.add_argument('--csv', type=str, default=None, help='good_covers_for_lora.csv yolu')
    parser.add_argument('--prefix', type=str, default='book cover, bestseller, ', help='Caption prefix')

    args = parser.parse_args()

    print("="*60)
    print("CAPTION ÜRETİCİ")
    print("="*60)

    if not os.path.exists(args.input_dir):
        print(f"HATA: Klasör bulunamadı: {args.input_dir}")
        sys.exit(1)

    if args.mode == 'blip':
        if not check_dependencies():
            print("Paketler yüklendi. Scripti tekrar çalıştırın.")
            sys.exit(0)
        generate_captions_blip(args.input_dir, args.prefix)
    else:
        csv_path = args.csv or os.path.join(BASE_DIR, "data", "processed", "good_covers_for_lora.csv")
        generate_simple_captions(args.input_dir, csv_path)

    print("\nTamamlandı!")


if __name__ == "__main__":
    main()
