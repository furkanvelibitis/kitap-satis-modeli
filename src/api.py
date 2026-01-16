"""
Kitap Satış Tahmin API
FastAPI ile REST API

Kullanım:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict - Kapak + bilgilerle tahmin
    POST /analyze-cover - Sadece kapak analizi
    GET /ideal-ranges/{publisher} - Yayınevi için ideal aralıklar
    GET /health - Sağlık kontrolü
    POST /generate-cover - LoRA ile kapak üretimi
    GET /lora-status - LoRA model durumu
"""

import os
import sys
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import pickle

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from model import TwoStageBookModel, get_val_transforms
from predict import BookSalesPredictor

# FastAPI app
app = FastAPI(
    title="Kitap Satış Tahmin API",
    description="Kitap kapağı ve meta verilerden satış tahmini",
    version="1.0.0"
)

# CORS - tüm originlere izin ver (production'da kısıtla)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (lazy loading)
predictor = None
ideal_ranges = None
lora_pipeline = None
lora_loaded = False


def get_predictor():
    """Lazy load predictor"""
    global predictor
    if predictor is None:
        print("Model yükleniyor...")
        predictor = BookSalesPredictor()
        print("Model hazır!")
    return predictor


def get_ideal_ranges():
    """Lazy load ideal ranges"""
    global ideal_ranges
    if ideal_ranges is None:
        path = os.path.join(BASE_DIR, "data", "processed", "ideal_ranges.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ideal_ranges = pickle.load(f)
    return ideal_ranges


def get_lora_pipeline():
    """Lazy load LoRA pipeline"""
    global lora_pipeline, lora_loaded

    if lora_pipeline is not None:
        return lora_pipeline

    lora_dir = os.path.join(BASE_DIR, "models", "lora_sdxl", "final")

    if not os.path.exists(lora_dir):
        return None

    try:
        print("LoRA modeli yükleniyor... (bu biraz zaman alabilir)")
        from diffusers import StableDiffusionXLPipeline
        from peft import PeftModel

        # Base SDXL modelini yükle
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )

        # LoRA ağırlıklarını yükle
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)

        # GPU'ya taşı
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            # Memory optimization
            pipe.enable_attention_slicing()

        lora_pipeline = pipe
        lora_loaded = True
        print("LoRA modeli hazır!")
        return pipe

    except Exception as e:
        print(f"LoRA yükleme hatası: {e}")
        return None


# Request/Response Models
class PredictionRequest(BaseModel):
    title: str
    author: str
    publisher: str
    page: int
    price: float
    year: Optional[int] = 2024
    discount_rate: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    kitap_adi: str
    yazar: str
    yayinevi: str
    baseline_yorum: int
    kapak_etkisi: float
    kapak_etkisi_pct: str
    kapak_degerlendirme: str
    tahmini_yorum: int
    tahmini_satis: int
    basari_kategorisi: str
    ideal_araliklar: Optional[dict] = None


class CoverAnalysisResponse(BaseModel):
    kapak_etkisi: float
    yorum: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None


class LoraStatusResponse(BaseModel):
    lora_available: bool
    lora_loaded: bool
    lora_path: str
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None


class GenerateCoverResponse(BaseModel):
    success: bool
    image_base64: Optional[str] = None
    prompt_used: str
    error: Optional[str] = None


# Endpoints
@app.get("/", tags=["Info"])
async def root():
    """API bilgisi"""
    return {
        "name": "Kitap Satış Tahmin API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Tam tahmin (kapak + meta)",
            "POST /analyze-cover": "Sadece kapak analizi",
            "GET /ideal-ranges/{publisher}": "Yayınevi ideal aralıkları",
            "GET /health": "Sağlık kontrolü"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Sistem sağlık kontrolü"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(...),
    publisher: str = Form(...),
    page: int = Form(...),
    price: float = Form(...),
    year: int = Form(2024),
    discount_rate: float = Form(0.0)
):
    """
    Kitap satış tahmini

    - **image**: Kapak resmi (jpg/png)
    - **title**: Kitap adı
    - **author**: Yazar adı
    - **publisher**: Yayınevi
    - **page**: Sayfa sayısı
    - **price**: Fiyat (TL)
    - **year**: Yayın yılı (opsiyonel)
    - **discount_rate**: İndirim oranı 0-1 (opsiyonel)
    """
    try:
        pred = get_predictor()

        # Resmi oku
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Geçici dosyaya kaydet (predictor dosya yolu bekliyor)
        temp_path = "/tmp/temp_cover.jpg"
        img.save(temp_path)

        # Tahmin yap
        result = pred.predict(
            image_path=temp_path,
            title=title,
            author=author,
            publisher=publisher,
            page=page,
            price=price,
            year=year,
            discount_rate=discount_rate
        )

        # İdeal aralıkları ekle
        ideal = pred.get_ideal_ranges(publisher, page, price)

        # Temizlik
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return PredictionResponse(
            kitap_adi=result['kitap_adi'],
            yazar=result['yazar'],
            yayinevi=result['yayinevi'],
            baseline_yorum=result['baseline_yorum'],
            kapak_etkisi=result['kapak_etkisi'],
            kapak_etkisi_pct=result['kapak_etkisi_pct'],
            kapak_degerlendirme=result['kapak_degerlendirme'],
            tahmini_yorum=result['tahmini_yorum'],
            tahmini_satis=result['tahmini_satis'],
            basari_kategorisi=result['basari_kategorisi'],
            ideal_araliklar=ideal
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-cover", response_model=CoverAnalysisResponse, tags=["Prediction"])
async def analyze_cover(
    image: UploadFile = File(...),
    publisher: str = Form("Genel"),
    author: str = Form("Bilinmiyor")
):
    """
    Sadece kapak analizi (hızlı)

    - **image**: Kapak resmi
    - **publisher**: Yayınevi (segment için)
    - **author**: Yazar (segment için)
    """
    try:
        pred = get_predictor()

        # Resmi oku
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        temp_path = "/tmp/temp_cover.jpg"
        img.save(temp_path)

        result = pred.analyze_cover_only(temp_path, publisher, author)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return CoverAnalysisResponse(
            kapak_etkisi=result['kapak_etkisi'],
            yorum=result['yorum']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ideal-ranges/{publisher}", tags=["Analysis"])
async def get_publisher_ideal_ranges(publisher: str):
    """
    Yayınevi için ideal fiyat/sayfa aralıkları

    - **publisher**: Yayınevi adı
    """
    ranges = get_ideal_ranges()

    if ranges is None:
        raise HTTPException(status_code=404, detail="İdeal aralıklar yüklenmemiş")

    publisher_ideals = ranges.get('publisher_ideals', {})

    if publisher in publisher_ideals:
        return {
            "publisher": publisher,
            "ideals": publisher_ideals[publisher],
            "found": True
        }
    else:
        # Global istatistikleri döndür
        return {
            "publisher": publisher,
            "ideals": ranges.get('global_stats', {}),
            "found": False,
            "message": "Yayınevi bulunamadı, genel istatistikler döndürüldü"
        }


@app.get("/publishers", tags=["Analysis"])
async def list_publishers():
    """Mevcut yayınevlerini listele"""
    ranges = get_ideal_ranges()

    if ranges is None:
        raise HTTPException(status_code=404, detail="İdeal aralıklar yüklenmemiş")

    publishers = list(ranges.get('publisher_ideals', {}).keys())
    return {
        "count": len(publishers),
        "publishers": sorted(publishers)[:100],  # İlk 100
        "message": f"Toplam {len(publishers)} yayınevi"
    }


@app.post("/predict-base64", tags=["Prediction"])
async def predict_base64(
    image_base64: str = Form(...),
    title: str = Form(...),
    author: str = Form(...),
    publisher: str = Form(...),
    page: int = Form(...),
    price: float = Form(...)
):
    """
    Base64 encoded resim ile tahmin (web entegrasyonu için)
    """
    try:
        pred = get_predictor()

        # Base64 decode
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')

        temp_path = "/tmp/temp_cover.jpg"
        img.save(temp_path)

        result = pred.predict(
            image_path=temp_path,
            title=title,
            author=author,
            publisher=publisher,
            page=page,
            price=price
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# LORA KAPAK ÜRETİMİ ENDPOINTLERİ
# =============================================

@app.get("/lora-status", response_model=LoraStatusResponse, tags=["LoRA - Kapak Üretimi"])
async def lora_status():
    """
    LoRA model durumu kontrolü
    """
    lora_dir = os.path.join(BASE_DIR, "models", "lora_sdxl", "final")
    lora_available = os.path.exists(lora_dir)

    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return LoraStatusResponse(
        lora_available=lora_available,
        lora_loaded=lora_loaded,
        lora_path=lora_dir,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_gb=gpu_memory
    )


@app.post("/generate-cover", response_model=GenerateCoverResponse, tags=["LoRA - Kapak Üretimi"])
async def generate_cover(
    prompt: str = Form(default="book cover, bestseller, professional design"),
    negative_prompt: str = Form(default="blurry, low quality, text, watermark, signature"),
    num_inference_steps: int = Form(default=30, ge=10, le=50),
    guidance_scale: float = Form(default=7.5, ge=1.0, le=15.0),
    width: int = Form(default=768, ge=512, le=1024),
    height: int = Form(default=1024, ge=512, le=1024),
    seed: Optional[int] = Form(default=None)
):
    """
    LoRA modeli ile kitap kapağı üret

    - **prompt**: Kapak açıklaması (İngilizce önerilir)
    - **negative_prompt**: İstenmeyen özellikler
    - **num_inference_steps**: Üretim adımı (fazla = kaliteli ama yavaş)
    - **guidance_scale**: Prompt'a bağlılık (7-8 ideal)
    - **width**: Genişlik (768 önerilir)
    - **height**: Yükseklik (1024 önerilir - kitap kapağı oranı)
    - **seed**: Rastgelelik seed'i (aynı sonuç için)
    """
    try:
        pipe = get_lora_pipeline()

        if pipe is None:
            return GenerateCoverResponse(
                success=False,
                prompt_used=prompt,
                error="LoRA modeli bulunamadı. models/lora_sdxl/final klasörünü kontrol edin."
            )

        # Seed ayarla
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

        # Kapak üret
        print(f"Kapak üretiliyor: {prompt[:50]}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        # Base64'e çevir
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return GenerateCoverResponse(
            success=True,
            image_base64=img_base64,
            prompt_used=prompt
        )

    except Exception as e:
        return GenerateCoverResponse(
            success=False,
            prompt_used=prompt,
            error=str(e)
        )


@app.post("/generate-cover-for-book", tags=["LoRA - Kapak Üretimi"])
async def generate_cover_for_book(
    title: str = Form(...),
    author: str = Form(...),
    genre: str = Form(default="roman"),
    mood: str = Form(default="professional"),
    num_inference_steps: int = Form(default=30),
    seed: Optional[int] = Form(default=None)
):
    """
    Kitap bilgilerine göre otomatik prompt oluşturup kapak üret

    - **title**: Kitap adı
    - **author**: Yazar adı
    - **genre**: Tür (roman, polisiye, bilim kurgu, tarih, vb.)
    - **mood**: Atmosfer (professional, dark, colorful, minimalist, vb.)
    """
    try:
        pipe = get_lora_pipeline()

        if pipe is None:
            return {
                "success": False,
                "error": "LoRA modeli bulunamadı"
            }

        # Türe göre prompt oluştur
        genre_prompts = {
            "roman": "literary fiction, elegant",
            "polisiye": "mystery, dark atmosphere, suspense",
            "bilim kurgu": "science fiction, futuristic, space",
            "fantastik": "fantasy, magical, mystical",
            "tarih": "historical, vintage, classic",
            "aşk": "romance, soft colors, emotional",
            "korku": "horror, dark, scary, gothic",
            "çocuk": "children book, colorful, playful, cartoon"
        }

        genre_desc = genre_prompts.get(genre.lower(), "professional design")

        prompt = f"book cover, bestseller, {genre_desc}, {mood}, high quality, detailed"
        negative_prompt = "blurry, low quality, text, watermark, signature, amateur"

        # Seed ayarla
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

        print(f"Kapak üretiliyor: '{title}' by {author}")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            width=768,
            height=1024,
            generator=generator
        ).images[0]

        # Base64'e çevir
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "success": True,
            "title": title,
            "author": author,
            "genre": genre,
            "prompt_used": prompt,
            "image_base64": img_base64
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modeli yükle"""
    print("API başlatılıyor...")
    print(f"GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # LoRA model kontrolü
    lora_dir = os.path.join(BASE_DIR, "models", "lora_sdxl", "final")
    if os.path.exists(lora_dir):
        print(f"LoRA modeli bulundu: {lora_dir}")
    else:
        print("LoRA modeli bulunamadı (opsiyonel)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
