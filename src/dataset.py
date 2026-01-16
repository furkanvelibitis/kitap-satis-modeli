"""
PyTorch Dataset Sınıfları v2
İki aşamalı model için veri yükleme
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import pickle

from model import get_train_transforms, get_val_transforms


# Paths - Otomatik olarak src/ klasörünün parent'ını bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
COVERS_DIR = os.path.join(DATA_DIR, "covers")


class TwoStageDataset(Dataset):
    """
    İki Aşamalı Model için Dataset

    Her örnek için:
        - Kapak resmi
        - Baseline features (yazar, yayınevi, segment vs.)
        - Context features (segment ortalaması vs.)
        - Hedef 1: reviews_log (mutlak satış)
        - Hedef 2: residual (kapak etkisi)
    """

    def __init__(self, dataframe, baseline_features, context_features,
                 transform=None, covers_dir=COVERS_DIR):
        self.df = dataframe.reset_index(drop=True)
        self.baseline_features = baseline_features
        self.context_features = context_features
        self.transform = transform
        self.covers_dir = covers_dir

        # Baseline özellikleri
        self.baseline_data = self.df[baseline_features].fillna(0).values.astype(np.float32)

        # Context özellikleri
        self.context_data = self.df[context_features].fillna(0).values.astype(np.float32)

        # Hedef değişkenler
        self.targets_absolute = self.df['reviews_log'].values.astype(np.float32)
        self.targets_residual = self.df['residual'].values.astype(np.float32)
        self.baseline_predictions = self.df['baseline_prediction'].values.astype(np.float32)

        # ISBN listesi
        self.isbns = self.df['isbn'].values

        # İyi kapak flag'i (LoRA için)
        if 'is_truly_good_cover' in self.df.columns:
            self.is_good_cover = self.df['is_truly_good_cover'].values
        else:
            self.is_good_cover = np.zeros(len(self.df), dtype=bool)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Baseline features
        baseline = torch.tensor(self.baseline_data[idx], dtype=torch.float32)

        # Context features
        context = torch.tensor(self.context_data[idx], dtype=torch.float32)

        # Hedefler
        target_absolute = torch.tensor(self.targets_absolute[idx], dtype=torch.float32)
        target_residual = torch.tensor(self.targets_residual[idx], dtype=torch.float32)
        baseline_pred = torch.tensor(self.baseline_predictions[idx], dtype=torch.float32)

        # Kapak resmi
        isbn = self.isbns[idx]
        image_path = os.path.join(self.covers_dir, f"{isbn}.jpg")

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)

        return {
            'image': image,
            'baseline_features': baseline,
            'context_features': context,
            'target_absolute': target_absolute,
            'target_residual': target_residual,
            'baseline_prediction': baseline_pred,
            'isbn': str(isbn),
            'is_good_cover': self.is_good_cover[idx]
        }


class BaselineOnlyDataset(Dataset):
    """
    Sadece baseline model için dataset (kapak yok, hızlı)
    Aşama 1 eğitimi için kullanılır
    """

    def __init__(self, dataframe, baseline_features):
        self.df = dataframe.reset_index(drop=True)
        self.baseline_features = baseline_features

        self.baseline_data = self.df[baseline_features].fillna(0).values.astype(np.float32)
        self.targets = self.df['reviews_log'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        baseline = torch.tensor(self.baseline_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            'baseline_features': baseline,
            'target': target
        }


class ResidualDataset(Dataset):
    """
    Sadece kapak etkisi (residual) modeli için dataset
    Aşama 2 eğitimi için kullanılır (baseline dondurulmuş)
    """

    def __init__(self, dataframe, context_features, transform=None, covers_dir=COVERS_DIR):
        self.df = dataframe.reset_index(drop=True)
        self.context_features = context_features
        self.transform = transform
        self.covers_dir = covers_dir

        self.context_data = self.df[context_features].fillna(0).values.astype(np.float32)
        self.targets_residual = self.df['residual'].values.astype(np.float32)
        self.isbns = self.df['isbn'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = torch.tensor(self.context_data[idx], dtype=torch.float32)
        target_residual = torch.tensor(self.targets_residual[idx], dtype=torch.float32)

        isbn = self.isbns[idx]
        image_path = os.path.join(self.covers_dir, f"{isbn}.jpg")

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        return {
            'image': image,
            'context_features': context,
            'target_residual': target_residual,
            'isbn': str(isbn)
        }


def load_processed_data():
    """İşlenmiş veriyi yükle"""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    with open(os.path.join(PROCESSED_DIR, "feature_sets.pkl"), 'rb') as f:
        feature_sets = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "encoders.pkl"), 'rb') as f:
        encoders = pickle.load(f)

    return train_df, test_df, feature_sets, encoders


def create_two_stage_dataloaders(train_df, test_df, feature_sets, batch_size=16, num_workers=0):
    """
    İki aşamalı model için DataLoader'lar
    """
    baseline_features = feature_sets['baseline_features']
    context_features = feature_sets['context_features']

    # Datasets
    train_dataset = TwoStageDataset(
        train_df,
        baseline_features,
        context_features,
        transform=get_train_transforms()
    )

    test_dataset = TwoStageDataset(
        test_df,
        baseline_features,
        context_features,
        transform=get_val_transforms()
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def create_baseline_dataloaders(train_df, test_df, feature_sets, batch_size=64):
    """
    Sadece baseline model için DataLoader'lar (hızlı eğitim)
    """
    baseline_features = feature_sets['baseline_features']

    train_dataset = BaselineOnlyDataset(train_df, baseline_features)
    test_dataset = BaselineOnlyDataset(test_df, baseline_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


def create_residual_dataloaders(train_df, test_df, feature_sets, batch_size=16, num_workers=0):
    """
    Sadece residual model için DataLoader'lar (kapak etkisi)
    """
    context_features = feature_sets['context_features']

    train_dataset = ResidualDataset(
        train_df, context_features, transform=get_train_transforms()
    )
    test_dataset = ResidualDataset(
        test_df, context_features, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    print("Dataset test ediliyor...")

    try:
        train_df, test_df, feature_sets, encoders = load_processed_data()
        print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
        print(f"Baseline features: {feature_sets['baseline_features']}")
        print(f"Context features: {feature_sets['context_features']}")

        # Two-stage DataLoader test
        train_loader, test_loader = create_two_stage_dataloaders(
            train_df, test_df, feature_sets, batch_size=4
        )

        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  Baseline features: {batch['baseline_features'].shape}")
        print(f"  Context features: {batch['context_features'].shape}")
        print(f"  Target absolute: {batch['target_absolute'].shape}")
        print(f"  Target residual: {batch['target_residual'].shape}")

    except FileNotFoundError as e:
        print(f"Veri dosyası bulunamadı: {e}")
        print("Önce data_preparation.py çalıştırın.")
