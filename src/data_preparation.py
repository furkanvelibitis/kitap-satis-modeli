"""
Veri Hazırlama Modülü v2
Kategori bazlı normalizasyon ve residual learning için veri hazırlığı
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
import pickle
import sys
import io

# UTF-8 encoding (Windows icin)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass  # Linux'ta gerekli degil

# Paths - Otomatik olarak src/ klasörünün parent'ını bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
COVERS_DIR = os.path.join(DATA_DIR, "covers")


def load_raw_data():
    """Ham veriyi yükle"""
    print("Veri yükleniyor...")
    df = pd.read_csv(
        os.path.join(RAW_DIR, "books.csv"),
        encoding='utf-8',
        low_memory=False
    )
    print(f"  Toplam kayıt: {len(df):,}")
    return df


def clean_price(price_str):
    """Fiyat string'ini float'a çevir"""
    if pd.isna(price_str):
        return np.nan
    try:
        price_str = str(price_str).replace(',', '.').replace(' ', '')
        return float(price_str)
    except:
        return np.nan


def clean_discount_rate(rate_str):
    """İndirim oranını float'a çevir (0-1 arası)"""
    if pd.isna(rate_str):
        return 0.0
    try:
        rate_str = str(rate_str).replace('%', '').replace(' ', '')
        return float(rate_str) / 100
    except:
        return 0.0


def extract_year(date_str):
    """Tarihten yılı çıkar"""
    if pd.isna(date_str):
        return np.nan
    try:
        return int(str(date_str)[:4])
    except:
        return np.nan


def extract_category_from_link(link):
    """URL'den kategori çıkarmaya çalış (varsa)"""
    # Kitapyurdu URL'lerinde kategori bilgisi olabilir
    # Şimdilik basit bir yaklaşım - yayınevine göre implicit kategori
    return None


def clean_data(df):
    """Veriyi temizle ve dönüştür"""
    print("Veri temizleniyor...")

    df = df.copy()

    # 1. Fiyatları temizle
    df['price_clean'] = df['price'].apply(clean_price)
    df['discounted_price_clean'] = df['discounted_price'].apply(clean_price)
    df['discount_rate_clean'] = df['discount_rate'].apply(clean_discount_rate)

    # 2. Yılı çıkar
    df['year'] = df['date'].apply(extract_year)

    # 3. Sayfa sayısı - negatif veya aşırı değerleri filtrele
    df['page_clean'] = df['page'].apply(lambda x: x if 0 < x < 5000 else np.nan)

    # 4. Eksik değerleri doldur
    df['price_clean'] = df['price_clean'].fillna(df['price_clean'].median())
    df['page_clean'] = df['page_clean'].fillna(df['page_clean'].median())
    df['year'] = df['year'].fillna(df['year'].median())

    # 5. Hedef değişken: log transform
    df['reviews_log'] = np.log1p(df['reviews'])

    print(f"  Temizleme tamamlandı")
    return df


def create_publisher_features(df):
    """Yayınevi bazlı özellikler - GELİŞTİRİLMİŞ"""
    print("Yayınevi özellikleri oluşturuluyor...")

    # Yayınevi istatistikleri
    publisher_stats = df.groupby('publisher').agg({
        'reviews': ['mean', 'median', 'std', 'count', 'sum'],
        'reviews_log': ['mean', 'std']
    }).reset_index()
    publisher_stats.columns = [
        'publisher',
        'publisher_avg_reviews', 'publisher_median_reviews', 'publisher_std_reviews',
        'publisher_book_count', 'publisher_total_reviews',
        'publisher_avg_log', 'publisher_std_log'
    ]

    # Std için NaN'ları doldur (tek kitaplı yayınevleri)
    publisher_stats['publisher_std_reviews'] = publisher_stats['publisher_std_reviews'].fillna(0)
    publisher_stats['publisher_std_log'] = publisher_stats['publisher_std_log'].fillna(0)

    df = df.merge(publisher_stats, on='publisher', how='left')

    # Log transform
    df['publisher_avg_reviews_log'] = np.log1p(df['publisher_avg_reviews'])

    # Yayınevi "gücü" skoru
    df['publisher_power'] = np.log1p(df['publisher_total_reviews']) * np.log1p(df['publisher_book_count'])

    print(f"  {df['publisher'].nunique():,} farklı yayınevi")
    return df


def create_author_features(df):
    """Yazar bazlı özellikler - GELİŞTİRİLMİŞ"""
    print("Yazar özellikleri oluşturuluyor...")

    # Yazar istatistikleri
    author_stats = df.groupby('author').agg({
        'reviews': ['mean', 'median', 'std', 'count', 'sum', 'max'],
        'reviews_log': ['mean', 'std']
    }).reset_index()
    author_stats.columns = [
        'author',
        'author_avg_reviews', 'author_median_reviews', 'author_std_reviews',
        'author_book_count', 'author_total_reviews', 'author_max_reviews',
        'author_avg_log', 'author_std_log'
    ]

    # Std için NaN'ları doldur
    author_stats['author_std_reviews'] = author_stats['author_std_reviews'].fillna(0)
    author_stats['author_std_log'] = author_stats['author_std_log'].fillna(0)

    df = df.merge(author_stats, on='author', how='left')

    # Log transform
    df['author_avg_reviews_log'] = np.log1p(df['author_avg_reviews'])

    # Yazar "gücü" skoru - bestseller geçmişi önemli
    df['author_power'] = np.log1p(df['author_max_reviews']) * np.log1p(df['author_book_count'])

    # Yazar tutarlılığı (düşük std = tutarlı satış)
    df['author_consistency'] = 1 / (1 + df['author_std_log'])

    print(f"  {df['author'].nunique():,} farklı yazar")
    return df


def create_category_proxy(df):
    """
    Kategori proxy'si oluştur
    Elimizde direkt kategori yok, ama yayınevi + fiyat + sayfa kombinasyonundan
    implicit kategori çıkarabiliriz
    """
    print("Kategori proxy'si oluşturuluyor...")

    # Yayınevi bazlı "segment" - bazı yayınevleri belirli türlere odaklanır
    # Örn: Yapı Kredi = edebiyat, Pegasus = popüler, İş Bankası = klasik

    # Fiyat/sayfa oranı - kategori göstergesi olabilir
    df['price_per_page'] = df['price_clean'] / (df['page_clean'] + 1)

    # Sayfa grubu (kısa/orta/uzun kitap)
    df['page_group'] = pd.cut(
        df['page_clean'],
        bins=[0, 150, 300, 500, 5000],
        labels=['kisa', 'orta', 'uzun', 'cok_uzun']
    ).astype(str)

    # Fiyat grubu
    df['price_group'] = pd.cut(
        df['price_clean'],
        bins=[0, 30, 50, 80, 150, 1000],
        labels=['ucuz', 'ekonomik', 'orta', 'pahali', 'premium']
    ).astype(str)

    # Segment: yayınevi + fiyat grubu kombinasyonu
    # Bu bir proxy kategori görevi görür
    df['segment'] = df['publisher'].astype(str) + '_' + df['price_group']

    return df


def calculate_relative_scores(df):
    """
    KATEGORİ BAZLI NORMALİZASYON
    Her kitabın kendi segmentine göre ne kadar başarılı olduğunu hesapla
    """
    print("Relative (görece) skorlar hesaplanıyor...")

    # 1. Yayınevi bazlı relative score
    df['publisher_relative'] = (df['reviews_log'] - df['publisher_avg_log']) / (df['publisher_std_log'] + 0.1)

    # 2. Yazar bazlı relative score
    df['author_relative'] = (df['reviews_log'] - df['author_avg_log']) / (df['author_std_log'] + 0.1)

    # 3. Segment bazlı normalizasyon
    segment_stats = df.groupby('segment').agg({
        'reviews_log': ['mean', 'std']
    }).reset_index()
    segment_stats.columns = ['segment', 'segment_avg_log', 'segment_std_log']
    segment_stats['segment_std_log'] = segment_stats['segment_std_log'].fillna(0.1)

    df = df.merge(segment_stats, on='segment', how='left')
    df['segment_relative'] = (df['reviews_log'] - df['segment_avg_log']) / (df['segment_std_log'] + 0.1)

    # Relative skorları sınırla (-3 ile +3 arası, aşırı outlier'ları kırp)
    for col in ['publisher_relative', 'author_relative', 'segment_relative']:
        df[col] = df[col].clip(-3, 3)

    print("  Relative skorlar hazır")
    return df


def calculate_baseline_and_residual(df):
    """
    İKİ AŞAMALI MODEL İÇİN BASELINE VE RESİDUAL HESAPLA

    Baseline: Yazar + Yayınevi + Segment etkisi (kapak hariç her şey)
    Residual: Gerçek satış - Baseline = Kapağın etkisi
    """
    print("Baseline ve residual hesaplanıyor...")

    # Baseline için kullanılacak özellikler (kapak HARİÇ)
    baseline_features = [
        'author_avg_reviews_log',
        'author_book_count',
        'author_power',
        'author_consistency',
        'publisher_avg_reviews_log',
        'publisher_book_count',
        'publisher_power',
        'price_clean',
        'page_clean',
        'year',
        'discount_rate_clean',
        'rating'
    ]

    # Eksik değerleri doldur
    for col in baseline_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Mevcut özellikleri kullan
    available_baseline = [col for col in baseline_features if col in df.columns]

    X_baseline = df[available_baseline].values
    y = df['reviews_log'].values

    # Basit Ridge regression ile baseline tahmin
    # (Neural network eğitiminde de kullanılacak ama burası quick estimate için)
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_baseline, y)

    # Baseline tahminleri
    df['baseline_prediction'] = baseline_model.predict(X_baseline)

    # RESIDUAL = Gerçek - Baseline
    # Bu değer KAPAĞIN ETKİSİNİ temsil ediyor
    df['residual'] = df['reviews_log'] - df['baseline_prediction']

    # Residual istatistikleri
    print(f"  Baseline R²: {baseline_model.score(X_baseline, y):.4f}")
    print(f"  Residual mean: {df['residual'].mean():.4f}")
    print(f"  Residual std: {df['residual'].std():.4f}")

    # Baseline model'i kaydet
    with open(os.path.join(PROCESSED_DIR, "baseline_model.pkl"), 'wb') as f:
        pickle.dump({
            'model': baseline_model,
            'features': available_baseline
        }, f)

    return df, available_baseline


def calculate_segment_ideal_ranges(df):
    """
    SEGMENT BAZLI İDEAL FİYAT VE SAYFA ARALIĞI ANALİZİ

    Her segment için başarılı kitapların (yüksek reviews) fiyat ve sayfa
    dağılımlarını analiz eder.

    Returns:
        dict: Her segment için ideal aralıklar
    """
    print("Segment bazlı ideal aralıklar hesaplanıyor...")

    segment_ideals = {}

    # Her yayınevi için analiz
    for publisher in df['publisher'].unique():
        pub_df = df[df['publisher'] == publisher]

        if len(pub_df) < 5:  # Minimum 5 kitap olmalı
            continue

        # Başarılı kitaplar (üst %40)
        success_threshold = pub_df['reviews_log'].quantile(0.6)
        successful = pub_df[pub_df['reviews_log'] >= success_threshold]

        if len(successful) < 3:
            continue

        # Fiyat analizi
        price_stats = {
            'min': successful['price_clean'].quantile(0.1),
            'max': successful['price_clean'].quantile(0.9),
            'optimal': successful['price_clean'].median(),
            'mean': successful['price_clean'].mean()
        }

        # Sayfa analizi
        page_stats = {
            'min': successful['page_clean'].quantile(0.1),
            'max': successful['page_clean'].quantile(0.9),
            'optimal': successful['page_clean'].median(),
            'mean': successful['page_clean'].mean()
        }

        # Fiyat/sayfa oranı
        price_per_page = successful['price_per_page'].median()

        segment_ideals[publisher] = {
            'price': price_stats,
            'page': page_stats,
            'price_per_page': price_per_page,
            'sample_size': len(successful),
            'total_books': len(pub_df),
            'avg_reviews': pub_df['reviews'].mean(),
            'success_rate': len(successful) / len(pub_df)
        }

    # Fiyat grupları için de analiz
    price_group_ideals = {}
    for price_group in df['price_group'].unique():
        pg_df = df[df['price_group'] == price_group]

        if len(pg_df) < 10:
            continue

        success_threshold = pg_df['reviews_log'].quantile(0.6)
        successful = pg_df[pg_df['reviews_log'] >= success_threshold]

        if len(successful) < 5:
            continue

        price_group_ideals[price_group] = {
            'page_optimal': successful['page_clean'].median(),
            'page_min': successful['page_clean'].quantile(0.25),
            'page_max': successful['page_clean'].quantile(0.75),
            'avg_reviews': successful['reviews'].mean(),
            'sample_size': len(successful)
        }

    # Sayfa grupları için de analiz
    page_group_ideals = {}
    for page_group in df['page_group'].unique():
        pg_df = df[df['page_group'] == page_group]

        if len(pg_df) < 10:
            continue

        success_threshold = pg_df['reviews_log'].quantile(0.6)
        successful = pg_df[pg_df['reviews_log'] >= success_threshold]

        if len(successful) < 5:
            continue

        page_group_ideals[page_group] = {
            'price_optimal': successful['price_clean'].median(),
            'price_min': successful['price_clean'].quantile(0.25),
            'price_max': successful['price_clean'].quantile(0.75),
            'avg_reviews': successful['reviews'].mean(),
            'sample_size': len(successful)
        }

    # Global istatistikler
    global_stats = {
        'price': {
            'median': df['price_clean'].median(),
            'mean': df['price_clean'].mean(),
            'p25': df['price_clean'].quantile(0.25),
            'p75': df['price_clean'].quantile(0.75)
        },
        'page': {
            'median': df['page_clean'].median(),
            'mean': df['page_clean'].mean(),
            'p25': df['page_clean'].quantile(0.25),
            'p75': df['page_clean'].quantile(0.75)
        }
    }

    print(f"  {len(segment_ideals)} yayınevi için ideal aralıklar hesaplandı")
    print(f"  {len(price_group_ideals)} fiyat grubu analizi tamamlandı")
    print(f"  {len(page_group_ideals)} sayfa grubu analizi tamamlandı")

    return {
        'publisher_ideals': segment_ideals,
        'price_group_ideals': price_group_ideals,
        'page_group_ideals': page_group_ideals,
        'global_stats': global_stats
    }


def identify_good_covers(df, threshold_percentile=75):
    """
    LoRA EĞİTİMİ İÇİN İYİ KAPAKLARI BELİRLE

    İyi kapak = Yüksek pozitif residual
    (Yazar/yayınevi etkisi çıkarıldıktan sonra hala iyi satmış)
    """
    print(f"İyi kapaklar belirleniyor (residual > {threshold_percentile}. percentile)...")

    threshold = df['residual'].quantile(threshold_percentile / 100)

    df['is_good_cover'] = df['residual'] > threshold

    # Segment bazında da kontrol et
    # Her segmentte kendi ortalamasının üstündekiler
    df['is_segment_good'] = df['segment_relative'] > 0

    # Her iki kritere de uyanlar = gerçekten iyi kapak
    df['is_truly_good_cover'] = df['is_good_cover'] & df['is_segment_good']

    good_count = df['is_truly_good_cover'].sum()
    print(f"  Gerçekten iyi kapak: {good_count:,} ({100*good_count/len(df):.1f}%)")

    return df


def encode_categorical(df):
    """Kategorik değişkenleri encode et"""
    print("Kategorik değişkenler encode ediliyor...")

    encoders = {}

    # Cover type
    if 'cover' in df.columns:
        le_cover = LabelEncoder()
        df['cover_encoded'] = le_cover.fit_transform(df['cover'].fillna('Unknown'))
        encoders['cover'] = le_cover

    # Paper type
    if 'paper' in df.columns:
        le_paper = LabelEncoder()
        df['paper_encoded'] = le_paper.fit_transform(df['paper'].fillna('Unknown'))
        encoders['paper'] = le_paper

    # Language
    if 'language' in df.columns:
        le_lang = LabelEncoder()
        df['language_encoded'] = le_lang.fit_transform(df['language'].fillna('Unknown'))
        encoders['language'] = le_lang

    # Page group
    if 'page_group' in df.columns:
        le_page = LabelEncoder()
        df['page_group_encoded'] = le_page.fit_transform(df['page_group'].fillna('orta'))
        encoders['page_group'] = le_page

    # Price group
    if 'price_group' in df.columns:
        le_price = LabelEncoder()
        df['price_group_encoded'] = le_price.fit_transform(df['price_group'].fillna('orta'))
        encoders['price_group'] = le_price

    return df, encoders


def filter_with_covers(df, min_reviews=10):
    """Kapak resmi olan ve minimum yorum sayısına sahip kitapları filtrele"""
    print(f"Filtreleme (min {min_reviews} yorum + kapak resmi)...")

    # Minimum yorum filtresi
    df_filtered = df[df['reviews'] >= min_reviews].copy()
    print(f"  {min_reviews}+ yorumlu: {len(df_filtered):,}")

    # Kapak resmi var mı kontrol et
    def has_cover(isbn):
        cover_path = os.path.join(COVERS_DIR, f"{isbn}.jpg")
        return os.path.exists(cover_path)

    if os.path.exists(COVERS_DIR) and len(os.listdir(COVERS_DIR)) > 0:
        df_filtered['has_cover'] = df_filtered['isbn'].apply(has_cover)
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['has_cover'] == True]
        print(f"  Kapak resmi olan: {len(df_filtered):,} ({before - len(df_filtered):,} eksik)")
    else:
        print(f"  Kapak klasörü henüz yok/boş, filtre atlanıyor")
        df_filtered['has_cover'] = True

    return df_filtered


def prepare_feature_sets(df):
    """
    İKİ AŞAMALI MODEL İÇİN FEATURE SETLERI
    """
    print("Feature setleri hazırlanıyor...")

    # BASELINE FEATURES (Aşama 1 - kapak hariç)
    baseline_features = [
        'author_avg_reviews_log',
        'author_book_count',
        'author_power',
        'author_consistency',
        'publisher_avg_reviews_log',
        'publisher_book_count',
        'publisher_power',
        'price_clean',
        'page_clean',
        'year',
        'discount_rate_clean',
        'rating',
        'cover_encoded',
        'paper_encoded',
        'language_encoded',
        'page_group_encoded',
        'price_group_encoded'
    ]

    # CONTEXT FEATURES (Aşama 2 - kapak modeline ek bilgi olarak)
    context_features = [
        'segment_avg_log',  # Bu segmentte ortalama ne kadar satılır
        'publisher_avg_reviews_log',
        'author_avg_reviews_log'
    ]

    # Mevcut sütunları kullan
    available_baseline = [col for col in baseline_features if col in df.columns]
    available_context = [col for col in context_features if col in df.columns]

    print(f"  Baseline features: {len(available_baseline)}")
    print(f"  Context features: {len(available_context)}")

    # Scaling
    scaler_baseline = StandardScaler()
    X_baseline = scaler_baseline.fit_transform(df[available_baseline].fillna(0).values)

    scaler_context = StandardScaler()
    X_context = scaler_context.fit_transform(df[available_context].fillna(0).values)

    return {
        'baseline_features': available_baseline,
        'context_features': available_context,
        'scaler_baseline': scaler_baseline,
        'scaler_context': scaler_context
    }


def create_train_test_split(df, test_size=0.2, random_state=42):
    """Train/test split - stratified by success level"""
    print(f"Train/test split ({test_size*100:.0f}% test)...")

    # Stratification için residual'a göre grupla
    df['success_bucket'] = pd.qcut(
        df['residual'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['success_bucket']
    )

    print(f"  Train: {len(train_df):,}")
    print(f"  Test: {len(test_df):,}")

    return train_df, test_df


def save_processed_data(train_df, test_df, feature_sets, encoders, ideal_ranges=None):
    """İşlenmiş veriyi kaydet"""
    print("Veriler kaydediliyor...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # DataFrames
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False, encoding='utf-8-sig')
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False, encoding='utf-8-sig')

    # Feature sets ve scalers
    with open(os.path.join(PROCESSED_DIR, "feature_sets.pkl"), 'wb') as f:
        pickle.dump(feature_sets, f)

    # Encoders
    with open(os.path.join(PROCESSED_DIR, "encoders.pkl"), 'wb') as f:
        pickle.dump(encoders, f)

    # İdeal aralıklar
    if ideal_ranges:
        with open(os.path.join(PROCESSED_DIR, "ideal_ranges.pkl"), 'wb') as f:
            pickle.dump(ideal_ranges, f)
        print(f"  İdeal aralıklar kaydedildi")

    # İyi kapaklar listesi (LoRA için)
    good_covers_train = train_df[train_df['is_truly_good_cover'] == True][['isbn', 'title', 'author', 'residual', 'segment']]
    good_covers_train.to_csv(os.path.join(PROCESSED_DIR, "good_covers_for_lora.csv"), index=False, encoding='utf-8-sig')

    print(f"  Kaydedildi: {PROCESSED_DIR}")
    print(f"  LoRA için iyi kapak sayısı: {len(good_covers_train):,}")


def main():
    print("="*60)
    print("VERİ HAZIRLAMA v2 - İKİ AŞAMALI MODEL")
    print("="*60)
    print()

    # 1. Veriyi yükle
    df = load_raw_data()

    # 2. Temizle
    df = clean_data(df)

    # 3. Feature engineering
    df = create_publisher_features(df)
    df = create_author_features(df)
    df = create_category_proxy(df)
    df = calculate_relative_scores(df)
    df, encoders = encode_categorical(df)

    # 4. Filtrele (10+ yorum ve kapak resmi olanlar)
    df_filtered = filter_with_covers(df, min_reviews=10)

    # 5. Baseline ve Residual hesapla
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_filtered, baseline_features = calculate_baseline_and_residual(df_filtered)

    # 6. İyi kapakları belirle (LoRA için)
    df_filtered = identify_good_covers(df_filtered, threshold_percentile=75)

    # 7. Segment bazlı ideal aralıkları hesapla
    ideal_ranges = calculate_segment_ideal_ranges(df_filtered)

    # 8. Feature setleri hazırla
    feature_sets = prepare_feature_sets(df_filtered)

    # 9. Train/test split
    train_df, test_df = create_train_test_split(df_filtered)

    # 10. Kaydet
    save_processed_data(train_df, test_df, feature_sets, encoders, ideal_ranges)

    # 11. Özet
    print()
    print("="*60)
    print("TAMAMLANDI!")
    print("="*60)
    print(f"Train set: {len(train_df):,} kitap")
    print(f"Test set: {len(test_df):,} kitap")
    print()
    print("HEDEF DEĞİŞKENLER:")
    print(f"  reviews_log: Mutlak satış tahmini için")
    print(f"  residual: Kapak etkisi tahmini için (YENİ!)")
    print(f"  baseline_prediction: Yazar/yayınevi baseline'ı")
    print()
    print("İDEAL ARALIKLAR:")
    print(f"  Yayınevi bazlı: {len(ideal_ranges['publisher_ideals'])} yayınevi")
    print(f"  Fiyat grubu bazlı: {len(ideal_ranges['price_group_ideals'])} grup")
    print(f"  Sayfa grubu bazlı: {len(ideal_ranges['page_group_ideals'])} grup")
    print(f"  Dosya: {os.path.join(PROCESSED_DIR, 'ideal_ranges.pkl')}")
    print()
    print("LoRA EĞİTİMİ İÇİN:")
    print(f"  İyi kapak sayısı: {train_df['is_truly_good_cover'].sum():,}")
    print(f"  Dosya: {os.path.join(PROCESSED_DIR, 'good_covers_for_lora.csv')}")


if __name__ == "__main__":
    main()
