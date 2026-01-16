"""
LoRA Eğitim Scripti - Kitap Kapağı Üretici
SDXL tabanlı LoRA fine-tuning

Kullanım:
    python train_lora.py --data_dir ../data/covers --output_dir ../models/lora
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
COVERS_DIR = os.path.join(DATA_DIR, "covers")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    required = ['diffusers', 'transformers', 'accelerate', 'peft']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Eksik paketler: {missing}")
        print("Yükleniyor...")
        os.system(f"pip install {' '.join(missing)} bitsandbytes xformers")
        print("Paketler yüklendi. Scripti tekrar çalıştırın.")
        sys.exit(0)


def prepare_dataset(good_covers_csv, covers_dir, output_dir, max_images=500):
    """
    LoRA eğitimi için dataset hazırla

    Args:
        good_covers_csv: İyi kapakların listesi (data_preparation.py çıktısı)
        covers_dir: Kapak resimlerinin klasörü
        output_dir: Çıktı klasörü
        max_images: Maksimum resim sayısı
    """
    import pandas as pd
    import shutil
    from PIL import Image

    print("Dataset hazırlanıyor...")

    # İyi kapakları yükle
    df = pd.read_csv(good_covers_csv)

    # En yüksek residual'a sahip kapakları seç
    df_sorted = df.sort_values('residual', ascending=False).head(max_images)

    # Klasör oluştur
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # Resimleri kopyala ve caption oluştur
    copied = 0
    for _, row in df_sorted.iterrows():
        isbn = row['isbn']
        src_path = os.path.join(covers_dir, f"{isbn}.jpg")

        if os.path.exists(src_path):
            # Resmi kontrol et ve kopyala
            try:
                img = Image.open(src_path)
                if img.size[0] >= 256 and img.size[1] >= 256:
                    # Resmi kopyala
                    dst_path = os.path.join(train_dir, f"{isbn}.jpg")
                    shutil.copy(src_path, dst_path)

                    # Caption dosyası oluştur
                    caption = "book cover, bestseller, professional design, high quality"
                    caption_path = os.path.join(train_dir, f"{isbn}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                    copied += 1
            except Exception as e:
                continue

    print(f"  {copied} resim hazırlandı: {train_dir}")
    return train_dir


def train_lora_sdxl(
    train_dir,
    output_dir,
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    resolution=1024,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_train_steps=1500,
    lora_rank=32,
    save_steps=500
):
    """
    SDXL LoRA eğitimi - diffusers kullanarak
    """
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    from diffusers import DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset sınıfı
    class CoverDataset(Dataset):
        def __init__(self, data_dir, resolution=1024):
            self.data_dir = data_dir
            self.resolution = resolution

            # Resim dosyalarını bul
            self.images = []
            self.captions = []

            for f in os.listdir(data_dir):
                if f.endswith('.jpg') or f.endswith('.png'):
                    img_path = os.path.join(data_dir, f)
                    txt_path = os.path.join(data_dir, f.replace('.jpg', '.txt').replace('.png', '.txt'))

                    if os.path.exists(txt_path):
                        self.images.append(img_path)
                        with open(txt_path, 'r', encoding='utf-8') as file:
                            self.captions.append(file.read().strip())

            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            print(f"  Dataset: {len(self.images)} resim")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
            return {"image": image, "caption": self.captions[idx]}

    print("\nModel yükleniyor...")

    # VAE ve UNet yükle
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # UNet'e LoRA ekle
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=1e-2
    )

    # Dataset ve DataLoader
    dataset = CoverDataset(train_dir, resolution)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Training loop
    print("\nEğitim başlıyor...")

    unet.train()
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()

    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="Training")

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            images = batch["image"].to(device, dtype=torch.float16)
            captions = batch["caption"]

            # Latent'e encode et
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Noise ekle
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text encoding
            with torch.no_grad():
                text_input = tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

                text_input_2 = tokenizer_2(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                text_embeddings_2 = text_encoder_2(text_input_2.input_ids.to(device))[0]

                # SDXL pooled embeddings
                pooled_embeddings = text_encoder_2(text_input_2.input_ids.to(device), output_hidden_states=True).hidden_states[-2]

            # Time embeddings
            add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=torch.float16)
            add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

            # UNet prediction
            added_cond_kwargs = {"text_embeds": pooled_embeddings.mean(dim=1), "time_ids": add_time_ids}
            encoder_hidden_states = torch.cat([text_embeddings, text_embeddings_2], dim=-1)

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # Loss hesapla
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backward
            loss.backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

            # Checkpoint kaydet
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                unet.save_pretrained(checkpoint_dir)
                print(f"\n  Checkpoint kaydedildi: {checkpoint_dir}")

    # Final model kaydet
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    unet.save_pretrained(final_dir)
    print(f"\nEğitim tamamlandı! Model: {final_dir}")

    return final_dir


def generate_sample(lora_dir, prompt, output_path):
    """LoRA ile örnek resim üret"""
    from diffusers import StableDiffusionXLPipeline
    from peft import PeftModel

    print(f"\nÖrnek üretiliyor: {prompt}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # LoRA yükle
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe = pipe.to("cuda")

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(output_path)
    print(f"  Kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LoRA Eğitimi - Kitap Kapağı')
    parser.add_argument('--prepare_only', action='store_true', help='Sadece dataset hazırla')
    parser.add_argument('--max_images', type=int, default=300, help='Maksimum resim sayısı')
    parser.add_argument('--max_steps', type=int, default=1500, help='Eğitim adımı')
    parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank')
    parser.add_argument('--resolution', type=int, default=1024, help='Resim çözünürlüğü')
    parser.add_argument('--generate', action='store_true', help='Eğitim sonrası örnek üret')

    args = parser.parse_args()

    print("="*60)
    print("LORA EĞİTİMİ - KİTAP KAPAĞI ÜRETİCİ")
    print("="*60)

    # Bağımlılıkları kontrol et
    check_dependencies()

    # Paths
    good_covers_csv = os.path.join(PROCESSED_DIR, "good_covers_for_lora.csv")
    lora_data_dir = os.path.join(DATA_DIR, "lora_dataset")
    lora_output_dir = os.path.join(MODELS_DIR, "lora_sdxl")

    # Dataset hazırla
    if not os.path.exists(os.path.join(lora_data_dir, "train")):
        if not os.path.exists(good_covers_csv):
            print("HATA: good_covers_for_lora.csv bulunamadı!")
            print("Önce data_preparation.py çalıştırın.")
            sys.exit(1)

        train_dir = prepare_dataset(
            good_covers_csv,
            COVERS_DIR,
            lora_data_dir,
            max_images=args.max_images
        )
    else:
        train_dir = os.path.join(lora_data_dir, "train")
        print(f"Mevcut dataset kullanılıyor: {train_dir}")

    if args.prepare_only:
        print("\nDataset hazır. Eğitim için --prepare_only olmadan çalıştırın.")
        return

    # LoRA eğitimi
    os.makedirs(lora_output_dir, exist_ok=True)

    final_model = train_lora_sdxl(
        train_dir=train_dir,
        output_dir=lora_output_dir,
        resolution=args.resolution,
        max_train_steps=args.max_steps,
        lora_rank=args.lora_rank
    )

    # Örnek üret
    if args.generate:
        sample_path = os.path.join(lora_output_dir, "sample_cover.png")
        generate_sample(
            final_model,
            "book cover, bestseller, mystery novel, dark atmosphere, professional design",
            sample_path
        )

    print("\n" + "="*60)
    print("TAMAMLANDI!")
    print("="*60)
    print(f"LoRA modeli: {final_model}")
    print("\nKullanım:")
    print('  python train_lora.py --generate')
    print("\nVeya ComfyUI/Automatic1111'de LoRA olarak yükleyin.")


if __name__ == "__main__":
    main()
