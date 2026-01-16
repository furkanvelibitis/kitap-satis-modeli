"""
Tahmin Scripti v2 - İki Aşamalı Model
Yeni bir kitap için satış tahmini + kapak etkisi analizi

Kullanım:
    python predict.py --image kapak.jpg --title "Kitap Adı" --author "Yazar" --publisher "Yayınevi" --page 300 --price 50
"""

import os
import sys
import io
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import pickle

# UTF-8 encoding (Windows icin)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass  # Linux'ta gerekli degil

from model import TwoStageBookModel, get_val_transforms

# Paths - Otomatik olarak src/ klasörünün parent'ını bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


class BookSalesPredictor:
    """İki aşamalı kitap satış tahmini"""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature sets yükle
        self.feature_sets = self._load_feature_sets()

        # Model yükle
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "two_stage_model.pth")

        self.model = self._load_model(model_path)
        self.transform = get_val_transforms()

        # İstatistikleri yükle
        self.publisher_stats, self.author_stats, self.segment_stats = self._load_stats()
        self.encoders = self._load_encoders()

        # İdeal aralıkları yükle
        self.ideal_ranges = self._load_ideal_ranges()

    def _load_feature_sets(self):
        path = os.path.join(PROCESSED_DIR, "feature_sets.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_model(self, path):
        num_baseline = len(self.feature_sets['baseline_features'])
        num_context = len(self.feature_sets['context_features'])

        model = TwoStageBookModel(num_baseline, num_context)

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def _load_encoders(self):
        path = os.path.join(PROCESSED_DIR, "encoders.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_ideal_ranges(self):
        """Segment bazlı ideal fiyat/sayfa aralıklarını yükle"""
        path = os.path.join(PROCESSED_DIR, "ideal_ranges.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _load_stats(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "books.csv"), encoding='utf-8', low_memory=False)
        df['reviews_log'] = np.log1p(df['reviews'])

        # Publisher stats
        pub_stats = df.groupby('publisher').agg({
            'reviews': ['mean', 'count'],
            'reviews_log': 'mean'
        }).reset_index()
        pub_stats.columns = ['publisher', 'avg_reviews', 'book_count', 'avg_log']
        pub_stats = pub_stats.set_index('publisher').to_dict('index')

        # Author stats
        auth_stats = df.groupby('author').agg({
            'reviews': ['mean', 'count', 'max'],
            'reviews_log': 'mean'
        }).reset_index()
        auth_stats.columns = ['author', 'avg_reviews', 'book_count', 'max_reviews', 'avg_log']
        auth_stats = auth_stats.set_index('author').to_dict('index')

        # Segment stats (global averages for unknown segments)
        segment_stats = {
            'default_avg_log': df['reviews_log'].mean(),
            'default_avg_reviews': df['reviews'].mean()
        }

        return pub_stats, auth_stats, segment_stats

    def _prepare_baseline_features(self, author, publisher, page, price, year=2024,
                                    discount_rate=0, rating=0, cover='Karton Kapak',
                                    paper='2. Hm. Kağıt', language='TÜRKÇE'):
        """Baseline model için feature'lar"""

        # Publisher features
        if publisher in self.publisher_stats:
            pub = self.publisher_stats[publisher]
            pub_avg_log = np.log1p(pub['avg_reviews'])
            pub_count = pub['book_count']
            pub_power = np.log1p(pub['avg_reviews'] * pub['book_count'])
        else:
            pub_avg_log = np.log1p(20)
            pub_count = 10
            pub_power = np.log1p(200)

        # Author features
        if author in self.author_stats:
            auth = self.author_stats[author]
            auth_avg_log = np.log1p(auth['avg_reviews'])
            auth_count = auth['book_count']
            auth_max = auth['max_reviews']
            auth_power = np.log1p(auth_max) * np.log1p(auth_count)
            auth_consistency = 1.0
        else:
            auth_avg_log = np.log1p(15)
            auth_count = 3
            auth_power = np.log1p(50)
            auth_consistency = 0.5

        # Encode categorical
        def safe_encode(encoder_name, value, default=0):
            try:
                return self.encoders[encoder_name].transform([value])[0]
            except:
                return default

        # Price group
        if price < 30:
            price_group = 'ucuz'
        elif price < 50:
            price_group = 'ekonomik'
        elif price < 80:
            price_group = 'orta'
        elif price < 150:
            price_group = 'pahali'
        else:
            price_group = 'premium'

        # Page group
        if page < 150:
            page_group = 'kisa'
        elif page < 300:
            page_group = 'orta'
        elif page < 500:
            page_group = 'uzun'
        else:
            page_group = 'cok_uzun'

        # Feature dict
        features = {
            'author_avg_reviews_log': auth_avg_log,
            'author_book_count': auth_count,
            'author_power': auth_power,
            'author_consistency': auth_consistency,
            'publisher_avg_reviews_log': pub_avg_log,
            'publisher_book_count': pub_count,
            'publisher_power': pub_power,
            'price_clean': price,
            'page_clean': page,
            'year': year,
            'discount_rate_clean': discount_rate,
            'rating': rating,
            'cover_encoded': safe_encode('cover', cover),
            'paper_encoded': safe_encode('paper', paper),
            'language_encoded': safe_encode('language', language),
            'page_group_encoded': safe_encode('page_group', page_group),
            'price_group_encoded': safe_encode('price_group', price_group)
        }

        # Feature vector (sıralama önemli!)
        baseline_features = self.feature_sets['baseline_features']
        vector = [features.get(name, 0) for name in baseline_features]

        return np.array(vector, dtype=np.float32)

    def _prepare_context_features(self, publisher, author):
        """Context features (segment bilgisi)"""

        # Segment average
        if publisher in self.publisher_stats:
            segment_avg = self.publisher_stats[publisher]['avg_log']
        else:
            segment_avg = self.segment_stats['default_avg_log']

        # Publisher ve author log averages
        pub_avg = np.log1p(self.publisher_stats.get(publisher, {'avg_reviews': 20})['avg_reviews'])
        auth_avg = np.log1p(self.author_stats.get(author, {'avg_reviews': 15})['avg_reviews'])

        context_features = self.feature_sets['context_features']
        features = {
            'segment_avg_log': segment_avg,
            'publisher_avg_reviews_log': pub_avg,
            'author_avg_reviews_log': auth_avg
        }

        vector = [features.get(name, 0) for name in context_features]
        return np.array(vector, dtype=np.float32)

    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Resim yüklenemedi: {e}")
            return torch.zeros(3, 224, 224)

    @torch.no_grad()
    def predict(self, image_path, title, author, publisher, page, price, **kwargs):
        """
        Kitap satış tahmini yap

        Returns:
            dict: Tahmin sonuçları (baseline, kapak etkisi, final tahmin)
        """

        # Prepare inputs
        image = self._load_image(image_path).unsqueeze(0).to(self.device)
        baseline_feat = torch.tensor(
            self._prepare_baseline_features(author, publisher, page, price, **kwargs)
        ).unsqueeze(0).to(self.device)
        context_feat = torch.tensor(
            self._prepare_context_features(publisher, author)
        ).unsqueeze(0).to(self.device)

        # Predict
        final, baseline, cover_effect = self.model(
            image, baseline_feat, context_feat, return_components=True
        )

        final_log = final.item()
        baseline_log = baseline.item()
        cover_effect_val = cover_effect.item()

        # Convert to actual values
        final_reviews = np.expm1(final_log)
        baseline_reviews = np.expm1(baseline_log)

        # Estimated sales (reviews × multiplier)
        estimated_sales = final_reviews * 10

        # Cover effect interpretation
        if cover_effect_val > 0.5:
            cover_impact = "Çok Pozitif (+)"
        elif cover_effect_val > 0.1:
            cover_impact = "Pozitif"
        elif cover_effect_val > -0.1:
            cover_impact = "Nötr"
        elif cover_effect_val > -0.5:
            cover_impact = "Negatif"
        else:
            cover_impact = "Çok Negatif (-)"

        # Cover effect percentage
        cover_effect_pct = (np.expm1(baseline_log + cover_effect_val) / np.expm1(baseline_log) - 1) * 100

        return {
            'kitap_adi': title,
            'yazar': author,
            'yayinevi': publisher,
            # Baseline (yazar/yayınevi etkisi)
            'baseline_yorum': round(baseline_reviews),
            'baseline_log': round(baseline_log, 3),
            # Kapak etkisi
            'kapak_etkisi': round(cover_effect_val, 3),
            'kapak_etkisi_pct': f"{cover_effect_pct:+.1f}%",
            'kapak_degerlendirme': cover_impact,
            # Final tahmin
            'tahmini_yorum': round(final_reviews),
            'tahmini_satis': round(estimated_sales),
            'final_log': round(final_log, 3),
            # Kategori
            'basari_kategorisi': self._get_success_category(final_reviews)
        }

    def _get_success_category(self, reviews):
        if reviews < 10:
            return "Düşük (< 10 yorum)"
        elif reviews < 50:
            return "Orta (10-50 yorum)"
        elif reviews < 100:
            return "İyi (50-100 yorum)"
        elif reviews < 500:
            return "Çok İyi (100-500 yorum)"
        elif reviews < 1000:
            return "Başarılı (500-1000 yorum)"
        else:
            return "Best Seller (1000+ yorum)"

    def analyze_cover_only(self, image_path, publisher, author):
        """Sadece kapak etkisini analiz et (baseline olmadan)"""

        image = self._load_image(image_path).unsqueeze(0).to(self.device)
        context_feat = torch.tensor(
            self._prepare_context_features(publisher, author)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cover_effect = self.model.get_cover_effect(image, context_feat)

        return {
            'kapak_etkisi': round(cover_effect.item(), 3),
            'yorum': self._interpret_cover_effect(cover_effect.item())
        }

    def _interpret_cover_effect(self, effect):
        if effect > 0.8:
            return "Bu kapak satışları ciddi şekilde artırabilir!"
        elif effect > 0.3:
            return "Bu kapak satışlara pozitif katkı sağlar."
        elif effect > -0.3:
            return "Bu kapak satışları çok etkilemez (nötr)."
        elif effect > -0.8:
            return "Bu kapak satışları olumsuz etkileyebilir."
        else:
            return "Bu kapak satışları ciddi şekilde düşürebilir!"

    def get_ideal_ranges(self, publisher, page, price):
        """
        Yayınevi ve fiyat/sayfa grubu bazında ideal aralık önerileri

        Returns:
            dict: İdeal aralık önerileri ve mevcut değerlerin değerlendirmesi
        """
        if not self.ideal_ranges:
            return None

        recommendations = {}

        # Yayınevi bazlı öneriler
        pub_ideals = self.ideal_ranges.get('publisher_ideals', {})
        if publisher in pub_ideals:
            pub_data = pub_ideals[publisher]
            recommendations['publisher'] = {
                'price_optimal': round(pub_data['price']['optimal'], 2),
                'price_range': f"{pub_data['price']['min']:.0f} - {pub_data['price']['max']:.0f} TL",
                'page_optimal': round(pub_data['page']['optimal']),
                'page_range': f"{pub_data['page']['min']:.0f} - {pub_data['page']['max']:.0f}",
                'sample_size': pub_data['sample_size'],
                'avg_reviews': round(pub_data['avg_reviews'])
            }

            # Mevcut fiyat değerlendirmesi
            if pub_data['price']['min'] <= price <= pub_data['price']['max']:
                recommendations['price_assessment'] = "Uygun"
            elif price < pub_data['price']['min']:
                recommendations['price_assessment'] = f"Düşük (önerilen: {pub_data['price']['min']:.0f}+ TL)"
            else:
                recommendations['price_assessment'] = f"Yüksek (önerilen: max {pub_data['price']['max']:.0f} TL)"

            # Mevcut sayfa sayısı değerlendirmesi
            if pub_data['page']['min'] <= page <= pub_data['page']['max']:
                recommendations['page_assessment'] = "Uygun"
            elif page < pub_data['page']['min']:
                recommendations['page_assessment'] = f"Kısa (önerilen: {pub_data['page']['min']:.0f}+ sayfa)"
            else:
                recommendations['page_assessment'] = f"Uzun (önerilen: max {pub_data['page']['max']:.0f} sayfa)"

        # Fiyat grubu bazlı sayfa önerisi
        if price < 30:
            price_group = 'ucuz'
        elif price < 50:
            price_group = 'ekonomik'
        elif price < 80:
            price_group = 'orta'
        elif price < 150:
            price_group = 'pahali'
        else:
            price_group = 'premium'

        price_group_ideals = self.ideal_ranges.get('price_group_ideals', {})
        if price_group in price_group_ideals:
            pg_data = price_group_ideals[price_group]
            recommendations['price_group'] = {
                'group': price_group,
                'page_optimal': round(pg_data['page_optimal']),
                'page_range': f"{pg_data['page_min']:.0f} - {pg_data['page_max']:.0f}"
            }

        # Sayfa grubu bazlı fiyat önerisi
        if page < 150:
            page_group = 'kisa'
        elif page < 300:
            page_group = 'orta'
        elif page < 500:
            page_group = 'uzun'
        else:
            page_group = 'cok_uzun'

        page_group_ideals = self.ideal_ranges.get('page_group_ideals', {})
        if page_group in page_group_ideals:
            pgg_data = page_group_ideals[page_group]
            recommendations['page_group'] = {
                'group': page_group,
                'price_optimal': round(pgg_data['price_optimal'], 2),
                'price_range': f"{pgg_data['price_min']:.0f} - {pgg_data['price_max']:.0f} TL"
            }

        # Global istatistikler
        global_stats = self.ideal_ranges.get('global_stats', {})
        if global_stats:
            recommendations['global'] = {
                'median_price': round(global_stats['price']['median'], 2),
                'median_page': round(global_stats['page']['median'])
            }

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Kitap satış tahmini (v2)')
    parser.add_argument('--image', required=True, help='Kapak resmi yolu')
    parser.add_argument('--title', required=True, help='Kitap adı')
    parser.add_argument('--author', required=True, help='Yazar')
    parser.add_argument('--publisher', required=True, help='Yayınevi')
    parser.add_argument('--page', type=int, required=True, help='Sayfa sayısı')
    parser.add_argument('--price', type=float, required=True, help='Fiyat (TL)')
    parser.add_argument('--discount', type=float, default=0, help='İndirim oranı')
    parser.add_argument('--year', type=int, default=2024, help='Yayın yılı')

    args = parser.parse_args()

    print("="*60)
    print("KİTAP SATIŞ TAHMİN SİSTEMİ v2")
    print("İki Aşamalı Model (Baseline + Kapak Etkisi)")
    print("="*60)

    predictor = BookSalesPredictor()

    result = predictor.predict(
        image_path=args.image,
        title=args.title,
        author=args.author,
        publisher=args.publisher,
        page=args.page,
        price=args.price,
        discount_rate=args.discount,
        year=args.year
    )

    print(f"\n{result['kitap_adi']}")
    print(f"Yazar: {result['yazar']} | Yayinevi: {result['yayinevi']}")
    print()
    print("-"*50)
    print("ASAMA 1: BASELINE (Yazar + Yayinevi Etkisi)")
    print("-"*50)
    print(f"  Beklenen yorum: ~{result['baseline_yorum']:,}")
    print(f"  (Sadece yazar ve yayinevi itibarına gore)")
    print()
    print("-"*50)
    print("ASAMA 2: KAPAK ETKISI")
    print("-"*50)
    print(f"  Kapak skoru: {result['kapak_etkisi']}")
    print(f"  Etki: {result['kapak_etkisi_pct']}")
    print(f"  Degerlendirme: {result['kapak_degerlendirme']}")
    print()
    print("-"*50)
    print("FINAL TAHMIN (Baseline + Kapak)")
    print("-"*50)
    print(f"  Tahmini yorum: ~{result['tahmini_yorum']:,}")
    print(f"  Tahmini satis: ~{result['tahmini_satis']:,} adet")
    print(f"  Kategori: {result['basari_kategorisi']}")
    print()

    # İdeal aralık önerileri
    ideal_ranges = predictor.get_ideal_ranges(args.publisher, args.page, args.price)
    if ideal_ranges:
        print("-"*50)
        print("IDEAL ARALIK ONERILERI")
        print("-"*50)

        # Yayınevi bazlı öneriler
        if 'publisher' in ideal_ranges:
            pub = ideal_ranges['publisher']
            print(f"\n  [{args.publisher}] icin basarili kitaplar:")
            print(f"    Ideal fiyat: {pub['price_optimal']} TL ({pub['price_range']})")
            print(f"    Ideal sayfa: {pub['page_optimal']} ({pub['page_range']})")
            print(f"    Orneklem: {pub['sample_size']} basarili kitap")

        # Mevcut değerlerin değerlendirmesi
        if 'price_assessment' in ideal_ranges:
            print(f"\n  Mevcut fiyat ({args.price} TL): {ideal_ranges['price_assessment']}")
        if 'page_assessment' in ideal_ranges:
            print(f"  Mevcut sayfa ({args.page}): {ideal_ranges['page_assessment']}")

        # Fiyat grubu bazlı
        if 'price_group' in ideal_ranges:
            pg = ideal_ranges['price_group']
            print(f"\n  [{pg['group'].upper()}] fiyat grubu icin:")
            print(f"    Onerilen sayfa: {pg['page_optimal']} ({pg['page_range']})")

        # Sayfa grubu bazlı
        if 'page_group' in ideal_ranges:
            pgg = ideal_ranges['page_group']
            print(f"\n  [{pgg['group'].upper()}] sayfa grubu icin:")
            print(f"    Onerilen fiyat: {pgg['price_optimal']} TL ({pgg['price_range']})")

        print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("="*60)
        print("KİTAP SATIŞ TAHMİN SİSTEMİ v2")
        print("="*60)
        print()
        print("Kullanım:")
        print('  python predict.py --image kapak.jpg --title "Kitap" \\')
        print('    --author "Yazar" --publisher "Yayınevi" --page 300 --price 50')
        print()
        print("Örnek:")
        print('  python predict.py --image roman.jpg --title "Gece Yarısı" \\')
        print('    --author "Elif Şafak" --publisher "Doğan Kitap" --page 320 --price 65')
    else:
        main()
