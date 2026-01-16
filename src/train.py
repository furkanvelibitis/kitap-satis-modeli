"""
Model Eğitim Scripti v2 - İKİ AŞAMALI EĞİTİM

Aşama 1: Baseline model eğitimi (yazar, yayınevi etkisi)
Aşama 2: Kapak etkisi modeli eğitimi (residual learning)
Aşama 3: End-to-end fine-tuning (opsiyonel)
"""

import os
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

# UTF-8 encoding (Windows icin)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass  # Linux'ta gerekli degil

from model import TwoStageBookModel, BaselineModel, CoverEffectModel, TabularOnlyModel
from dataset import (
    load_processed_data,
    create_two_stage_dataloaders,
    create_baseline_dataloaders,
    create_residual_dataloaders
)

# Paths - Otomatik olarak src/ klasörünün parent'ını bul
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")


class TwoStageTrainer:
    """İki aşamalı model eğitimi"""

    def __init__(self, device):
        self.device = device
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': [], 'val_mae': []},
            'stage2': {'train_loss': [], 'val_loss': [], 'val_mae': []},
            'combined': {'train_loss': [], 'val_loss': [], 'val_mae': []}
        }

    def train_stage1_baseline(self, model, train_loader, val_loader, epochs=30, lr=1e-3):
        """
        AŞAMA 1: Baseline model eğitimi
        Sadece meta verilerle (yazar, yayınevi) satış tahmini
        """
        print("\n" + "="*60)
        print("AŞAMA 1: BASELINE MODEL EĞİTİMİ")
        print("="*60)
        print("Yazar ve yayınevi etkisini öğreniyor...")

        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                features = batch['baseline_features'].to(self.device)
                targets = batch['target'].to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['baseline_features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    outputs = model(features)
                    val_loss += criterion(outputs, targets).item()
                    val_mae += torch.abs(outputs - targets).mean().item()

            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)
            self.history['stage1']['val_mae'].append(val_mae)

            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f} | LR: {current_lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, "baseline_model.pth"))

        print(f"\nAşama 1 tamamlandı! Best Val Loss: {best_val_loss:.4f}")
        return model

    def train_stage2_cover_effect(self, model, train_loader, val_loader, epochs=30, lr=1e-4):
        """
        AŞAMA 2: Kapak etkisi modeli eğitimi
        Residual learning - kapağın baseline'dan sapmasını öğreniyor
        """
        print("\n" + "="*60)
        print("AŞAMA 2: KAPAK ETKİSİ MODELİ EĞİTİMİ")
        print("="*60)
        print("Kapağın satışa etkisini öğreniyor (residual)...")

        model = model.to(self.device)
        criterion = nn.MSELoss()

        # Huber loss da kullanılabilir - outlier'lara daha dayanıklı
        # criterion = nn.HuberLoss(delta=1.0)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images = batch['image'].to(self.device)
                context = batch['context_features'].to(self.device)
                targets = batch['target_residual'].to(self.device)

                optimizer.zero_grad()
                outputs = model(images, context)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    context = batch['context_features'].to(self.device)
                    targets = batch['target_residual'].to(self.device)
                    outputs = model(images, context)
                    val_loss += criterion(outputs, targets).item()
                    val_mae += torch.abs(outputs - targets).mean().item()

            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            self.history['stage2']['train_loss'].append(train_loss)
            self.history['stage2']['val_loss'].append(val_loss)
            self.history['stage2']['val_mae'].append(val_mae)

            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f} | LR: {current_lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, "cover_effect_model.pth"))

        print(f"\nAşama 2 tamamlandı! Best Val Loss: {best_val_loss:.4f}")
        return model

    def train_combined_model(self, model, train_loader, val_loader, epochs=20, lr=5e-5):
        """
        AŞAMA 3 (OPSİYONEL): End-to-end fine-tuning
        Her iki aşamayı birlikte fine-tune et
        """
        print("\n" + "="*60)
        print("AŞAMA 3: END-TO-END FINE-TUNING")
        print("="*60)

        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images = batch['image'].to(self.device)
                baseline_feat = batch['baseline_features'].to(self.device)
                context_feat = batch['context_features'].to(self.device)
                targets = batch['target_absolute'].to(self.device)

                optimizer.zero_grad()
                outputs = model(images, baseline_feat, context_feat)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    baseline_feat = batch['baseline_features'].to(self.device)
                    context_feat = batch['context_features'].to(self.device)
                    targets = batch['target_absolute'].to(self.device)
                    outputs = model(images, baseline_feat, context_feat)
                    val_loss += criterion(outputs, targets).item()
                    val_mae += torch.abs(outputs - targets).mean().item()

            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            scheduler.step()

            self.history['combined']['train_loss'].append(train_loss)
            self.history['combined']['val_loss'].append(val_loss)
            self.history['combined']['val_mae'].append(val_mae)

            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'history': self.history
                }, os.path.join(MODELS_DIR, "two_stage_model.pth"))

        print(f"\nAşama 3 tamamlandı! Best Val Loss: {best_val_loss:.4f}")
        return model

    @torch.no_grad()
    def evaluate(self, model, test_loader, return_predictions=False):
        """Model değerlendirmesi"""
        model.eval()
        all_preds = []
        all_targets = []
        all_baselines = []
        all_residuals = []

        for batch in test_loader:
            images = batch['image'].to(self.device)
            baseline_feat = batch['baseline_features'].to(self.device)
            context_feat = batch['context_features'].to(self.device)
            targets = batch['target_absolute']

            final, baseline, residual = model(
                images, baseline_feat, context_feat, return_components=True
            )

            all_preds.extend(final.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_baselines.extend(baseline.cpu().numpy())
            all_residuals.extend(residual.cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        baselines = np.array(all_baselines)
        residuals = np.array(all_residuals)

        # Metrikleri hesapla
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)

        # Gerçek ölçekte
        pred_reviews = np.expm1(preds)
        true_reviews = np.expm1(targets)
        mae_real = np.mean(np.abs(pred_reviews - true_reviews))

        # Baseline ve cover effect ayrı analiz
        baseline_mae = np.mean(np.abs(baselines - targets))
        residual_contribution = np.std(residuals)

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mae_real': mae_real,
            'baseline_mae': baseline_mae,
            'residual_std': residual_contribution
        }

        if return_predictions:
            return results, preds, targets, baselines, residuals
        return results


def plot_training_history(history, save_path=None):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Stage 1
    if history['stage1']['val_loss']:
        axes[0].plot(history['stage1']['train_loss'], label='Train', alpha=0.7)
        axes[0].plot(history['stage1']['val_loss'], label='Val', alpha=0.7)
        axes[0].set_title('Aşama 1: Baseline Model')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Stage 2
    if history['stage2']['val_loss']:
        axes[1].plot(history['stage2']['train_loss'], label='Train', alpha=0.7)
        axes[1].plot(history['stage2']['val_loss'], label='Val', alpha=0.7)
        axes[1].set_title('Aşama 2: Kapak Etkisi')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Combined
    if history['combined']['val_loss']:
        axes[2].plot(history['combined']['train_loss'], label='Train', alpha=0.7)
        axes[2].plot(history['combined']['val_loss'], label='Val', alpha=0.7)
        axes[2].set_title('Aşama 3: Combined')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafik kaydedildi: {save_path}")

    plt.show()


def analyze_cover_effect(model, test_loader, device, save_path=None):
    """Kapak etkisini analiz et"""
    model.eval()

    cover_effects = []
    targets = []
    baselines = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            baseline_feat = batch['baseline_features'].to(device)
            context_feat = batch['context_features'].to(device)

            final, baseline, residual = model(
                images, baseline_feat, context_feat, return_components=True
            )

            cover_effects.extend(residual.cpu().numpy())
            targets.extend(batch['target_absolute'].numpy())
            baselines.extend(baseline.cpu().numpy())

    cover_effects = np.array(cover_effects)
    targets = np.array(targets)
    baselines = np.array(baselines)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cover effect dağılımı
    axes[0].hist(cover_effects, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', label='Nötr')
    axes[0].set_title('Kapak Etkisi Dağılımı')
    axes[0].set_xlabel('Cover Effect (residual)')
    axes[0].set_ylabel('Frekans')
    axes[0].legend()

    # Baseline vs Target
    axes[1].scatter(baselines, targets, alpha=0.3, s=10)
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[1].set_title('Baseline vs Gerçek')
    axes[1].set_xlabel('Baseline Tahmin')
    axes[1].set_ylabel('Gerçek Değer')

    # Cover effect vs Error
    errors = targets - baselines  # Baseline hatası = kapağın düzeltmesi gereken
    axes[2].scatter(errors, cover_effects, alpha=0.3, s=10)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('Baseline Hatası vs Kapak Etkisi')
    axes[2].set_xlabel('Baseline Hatası')
    axes[2].set_ylabel('Kapak Etkisi (tahmin)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()

    # İstatistikler
    print("\nKAPAK ETKİSİ ANALİZİ:")
    print(f"  Ortalama: {cover_effects.mean():.4f}")
    print(f"  Std: {cover_effects.std():.4f}")
    print(f"  Pozitif etki: {(cover_effects > 0).sum():,} ({100*(cover_effects > 0).mean():.1f}%)")
    print(f"  Negatif etki: {(cover_effects < 0).sum():,} ({100*(cover_effects < 0).mean():.1f}%)")


def main():
    print("="*60)
    print("KİTAP SATIŞ TAHMİN MODELİ v2")
    print("İKİ AŞAMALI EĞİTİM")
    print("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Veri yükle
    print("\nVeri yükleniyor...")
    train_df, test_df, feature_sets, encoders = load_processed_data()
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    num_baseline_features = len(feature_sets['baseline_features'])
    num_context_features = len(feature_sets['context_features'])
    print(f"Baseline features: {num_baseline_features}")
    print(f"Context features: {num_context_features}")

    # Trainer
    trainer = TwoStageTrainer(device)

    # ========================================
    # AŞAMA 1: Baseline Model
    # ========================================
    baseline_train_loader, baseline_val_loader = create_baseline_dataloaders(
        train_df, test_df, feature_sets, batch_size=64
    )

    baseline_model = BaselineModel(num_baseline_features)
    baseline_model = trainer.train_stage1_baseline(
        baseline_model, baseline_train_loader, baseline_val_loader,
        epochs=30, lr=1e-3
    )

    # ========================================
    # AŞAMA 2: Kapak Etkisi Modeli
    # ========================================
    residual_train_loader, residual_val_loader = create_residual_dataloaders(
        train_df, test_df, feature_sets, batch_size=16
    )

    cover_model = CoverEffectModel(context_dim=num_context_features)
    cover_model = trainer.train_stage2_cover_effect(
        cover_model, residual_train_loader, residual_val_loader,
        epochs=25, lr=1e-4
    )

    # ========================================
    # AŞAMA 3: Combined Fine-tuning
    # ========================================
    two_stage_train_loader, two_stage_val_loader = create_two_stage_dataloaders(
        train_df, test_df, feature_sets, batch_size=16
    )

    # Combined model oluştur ve ağırlıkları yükle
    combined_model = TwoStageBookModel(num_baseline_features, num_context_features)
    combined_model.baseline_model.load_state_dict(baseline_model.state_dict())
    combined_model.cover_model.load_state_dict(cover_model.state_dict())

    combined_model = trainer.train_combined_model(
        combined_model, two_stage_train_loader, two_stage_val_loader,
        epochs=15, lr=5e-5
    )

    # ========================================
    # DEĞERLENDİRME
    # ========================================
    print("\n" + "="*60)
    print("FİNAL DEĞERLENDİRME")
    print("="*60)

    results = trainer.evaluate(combined_model, two_stage_val_loader)

    print(f"\nLog Scale:")
    print(f"  MSE:  {results['mse']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"\nGerçek Ölçek:")
    print(f"  MAE:  {results['mae_real']:.1f} yorum")
    print(f"\nModel Bileşenleri:")
    print(f"  Baseline MAE: {results['baseline_mae']:.4f}")
    print(f"  Cover Effect Std: {results['residual_std']:.4f}")

    # Grafikler
    plot_training_history(trainer.history, os.path.join(MODELS_DIR, "training_history.png"))
    analyze_cover_effect(combined_model, two_stage_val_loader, device,
                         os.path.join(MODELS_DIR, "cover_effect_analysis.png"))

    # Sonuçları kaydet
    results['timestamp'] = datetime.now().isoformat()
    results['train_size'] = len(train_df)
    results['test_size'] = len(test_df)

    with open(os.path.join(MODELS_DIR, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSonuçlar kaydedildi: {MODELS_DIR}")


if __name__ == "__main__":
    main()
