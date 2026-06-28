"""
api/main.py
FastAPI demo server for DAFR-Net mural restoration.
Upload a damaged mural image → receive the restored output.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import yaml

from src.classifier.model import DamageClassifier
from src.encoder.dual_branch import DualBranchEncoder
from src.decoder.inpainting_head import InpaintingHead

app = FastAPI(
    title="DAFR-Net: Ancient Mural Restoration API",
    description="Damage-Aware Frequency-Guided Restoration Network — K Anirudh, SASTRA",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Load models on startup
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("configs/classifier.yaml") as f:
    clf_cfg = yaml.safe_load(f)

with open("configs/encoder.yaml") as f:
    enc_cfg = yaml.safe_load(f)

classifier = None
encoder    = None
decoder    = None


@app.on_event("startup")
def load_models():
    global classifier, encoder, decoder
    try:
        classifier = DamageClassifier.load_from_checkpoint(
            "models/exports/classifier.ckpt", config=clf_cfg
        ).to(DEVICE).eval()

        encoder = DualBranchEncoder(config=enc_cfg).to(DEVICE).eval()
        encoder.load_state_dict(
            torch.load("models/exports/encoder.pth", map_location=DEVICE)
        )

        decoder = InpaintingHead(in_channels=512).to(DEVICE).eval()
        decoder.load_state_dict(
            torch.load("models/exports/decoder.pth", map_location=DEVICE)
        )
        print(f"[DAFR-Net] Models loaded on {DEVICE}")
    except FileNotFoundError:
        print("[DAFR-Net] Checkpoints not found — run training first.")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.squeeze(0).clamp(-1, 1) + 1) / 2       # [-1,1] → [0,1]
    t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"project": "DAFR-Net", "author": "K Anirudh, SASTRA", "status": "Phase 1"}


@app.get("/health")
def health():
    return {
        "classifier_loaded": classifier is not None,
        "encoder_loaded":    encoder is not None,
        "decoder_loaded":    decoder is not None,
        "device": DEVICE,
    }


@app.post("/restore")
async def restore_mural(file: UploadFile = File(...)):
    """
    Upload a damaged mural image (PNG/JPEG).
    Returns the restored image as PNG.
    """
    if classifier is None or encoder is None or decoder is None:
        raise HTTPException(503, "Models not loaded. Run training first.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Step 1: classify damage
            damage_type = classifier.predict_damage_type(tensor)

            # Step 2: encode
            latent = encoder(tensor)

            # Step 3: decode
            restored = decoder(latent)

        result_img = tensor_to_pil(restored)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "X-Damage-Type": damage_type,
                "X-Model": "DAFR-Net Phase 1",
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Restoration failed: {str(e)}")


@app.post("/classify")
async def classify_damage(file: UploadFile = File(...)):
    """
    Classify the damage type of an uploaded mural image.
    Returns: crack | fade | missing
    """
    if classifier is None:
        raise HTTPException(503, "Classifier not loaded.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        damage_type = classifier.predict_damage_type(tensor)

    return {"damage_type": damage_type, "model": "DAFR-Net Classifier v1"}
