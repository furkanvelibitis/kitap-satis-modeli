"""
Kitap Satış Tahmin Modeli v2 - İKİ AŞAMALI MİMARİ

Aşama 1 (Baseline): Yazar + Yayınevi + Segment → Beklenen satış
Aşama 2 (Kapak Etkisi): Kapak resmi → Baseline'dan sapma (residual)

Final Tahmin = Baseline + Kapak Etkisi
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os


# ============================================================
# AŞAMA 1: BASELINE MODEL (Kapak Hariç Her Şey)
# ============================================================

class BaselineModel(nn.Module):
    """
    Yazar, yayınevi, segment gibi meta verilerden baseline satış tahmini

    Bu model "bu yazarın, bu yayınevinden çıkan kitabı ortalama ne satar?" sorusunu cevaplar.
    Kapak etkisi YOKTUR - sadece meta veriler.
    """

    def __init__(self, num_features, hidden_dims=[256, 128, 64]):
        super(BaselineModel, self).__init__()

        layers = []
        input_dim = num_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, num_features) - tablo özellikleri

        Returns:
            baseline_prediction: (batch,) - tahmini log(yorum sayısı)
        """
        return self.network(x).squeeze(1)


# ============================================================
# AŞAMA 2: KAPAK ETKİSİ MODELİ (Residual Learning)
# ============================================================

class CoverEffectModel(nn.Module):
    """
    Kapak resminden RESIDUAL (baseline'dan sapma) tahmini

    Bu model "bu kapak, beklenen satışa ne kadar ekler/çıkarır?" sorusunu cevaplar.

    Pozitif çıktı = Kapak satışı artırıyor
    Negatif çıktı = Kapak satışı düşürüyor
    """

    def __init__(self, context_dim=3, embedding_dim=256, pretrained=True):
        super(CoverEffectModel, self).__init__()

        # ResNet18 backbone (ImageNet pretrained)
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Image projection
        self.image_projection = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Context projection (segment bilgisi - ne bekleniyor?)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined prediction head
        # Kapak + Context → Residual
        self.residual_head = nn.Sequential(
            nn.Linear(embedding_dim + 64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, image, context):
        """
        Args:
            image: (batch, 3, 224, 224) - kapak resmi
            context: (batch, context_dim) - segment/kategori bilgisi

        Returns:
            residual: (batch,) - baseline'dan sapma tahmini
        """
        # Image encoding
        img_features = self.backbone(image)  # (batch, 512)
        img_embed = self.image_projection(img_features)  # (batch, 256)

        # Context encoding
        ctx_embed = self.context_projection(context)  # (batch, 64)

        # Combine and predict residual
        combined = torch.cat([img_embed, ctx_embed], dim=1)  # (batch, 320)
        residual = self.residual_head(combined)  # (batch, 1)

        return residual.squeeze(1)


# ============================================================
# BİRLEŞİK MODEL: İKİ AŞAMALI TAHMİN
# ============================================================

class TwoStageBookModel(nn.Module):
    """
    İki Aşamalı Kitap Satış Tahmin Modeli

    Final Tahmin = Baseline(meta) + CoverEffect(image, context)

    Bu yapı sayesinde:
    1. Tolstoy'un kitabı → yüksek baseline, kapak etkisi ayrı ölçülür
    2. Bilinmeyen yazarın kitabı → düşük baseline, iyi kapak pozitif etki yapar
    3. Model "kapağın gerçek katkısını" öğrenir
    """

    def __init__(self, num_baseline_features, num_context_features=3):
        super(TwoStageBookModel, self).__init__()

        self.baseline_model = BaselineModel(num_baseline_features)
        self.cover_model = CoverEffectModel(context_dim=num_context_features)

        # Learnable weight for combining (optional)
        self.combine_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, baseline_features, context_features, return_components=False):
        """
        Args:
            image: (batch, 3, 224, 224) - kapak resmi
            baseline_features: (batch, num_baseline_features) - meta veriler
            context_features: (batch, num_context_features) - segment bilgisi

        Returns:
            final_prediction: (batch,) - log(yorum sayısı) tahmini

            Eğer return_components=True:
                (final, baseline, residual) döner
        """
        # Aşama 1: Baseline tahmin
        baseline = self.baseline_model(baseline_features)

        # Aşama 2: Kapak etkisi (residual)
        residual = self.cover_model(image, context_features)

        # Final = Baseline + Kapak Etkisi
        final = baseline + self.combine_weight * residual

        if return_components:
            return final, baseline, residual
        return final

    def get_cover_effect(self, image, context_features):
        """Sadece kapak etkisini al (baseline olmadan)"""
        return self.cover_model(image, context_features)

    def get_baseline(self, baseline_features):
        """Sadece baseline tahminini al"""
        return self.baseline_model(baseline_features)


# ============================================================
# SADECE BASELINE MODELİ (Hızlı test için)
# ============================================================

class TabularOnlyModel(nn.Module):
    """Sadece tablo verisi kullanan model (baseline)"""

    def __init__(self, num_features):
        super(TabularOnlyModel, self).__init__()
        self.model = BaselineModel(num_features)

    def forward(self, tabular):
        return self.model(tabular)


# ============================================================
# SADECE KAPAK MODELİ (Ablation study için)
# ============================================================

class ImageOnlyModel(nn.Module):
    """Sadece kapak resmi kullanan model"""

    def __init__(self, pretrained=True):
        super(ImageOnlyModel, self).__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image):
        return self.backbone(image).squeeze(1)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

def get_train_transforms():
    """Eğitim için data augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """Validation/Test için transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path, transform=None):
    """Tek bir resmi yükle ve transform et"""
    try:
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        print(f"Resim yüklenemedi: {image_path}, Hata: {e}")
        return torch.zeros(3, 224, 224)


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary(model, model_name="Model"):
    """Model özetini yazdır"""
    print("="*60)
    print(f"MODEL: {model_name}")
    print("="*60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")

    # Alt modüllerin parametre sayıları
    if hasattr(model, 'baseline_model'):
        baseline_params = sum(p.numel() for p in model.baseline_model.parameters())
        print(f"  - Baseline model: {baseline_params:,}")

    if hasattr(model, 'cover_model'):
        cover_params = sum(p.numel() for p in model.cover_model.parameters())
        print(f"  - Cover model: {cover_params:,}")


def count_parameters(model):
    """Parametre sayısını döndür"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Model test ediliyor...")
    print()

    # Two-stage model
    model = TwoStageBookModel(
        num_baseline_features=17,
        num_context_features=3
    )
    print_model_summary(model, "TwoStageBookModel")

    # Dummy input
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_baseline = torch.randn(batch_size, 17)
    dummy_context = torch.randn(batch_size, 3)

    # Forward pass
    with torch.no_grad():
        final, baseline, residual = model(
            dummy_image, dummy_baseline, dummy_context,
            return_components=True
        )
        print(f"\nTest outputs:")
        print(f"  Final shape: {final.shape}")
        print(f"  Baseline: {baseline[:2].tolist()}")
        print(f"  Residual: {residual[:2].tolist()}")
        print(f"  Final: {final[:2].tolist()}")
