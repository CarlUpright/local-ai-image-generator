"""
Télécharge OpenDalle V1.1 dans le dossier models/ local.
Exécuter une seule fois avant de lancer app.py.

On télécharge uniquement les variantes fp16 (~7 GB au lieu de 27 GB).
Le chargement se fait ensuite avec variant="fp16" dans from_pretrained().
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

MODEL_ID = "dataautogpt3/OpenDalleV1.1"
MODEL_DIR = Path("models/OpenDalleV1.1")

# Fichier sentinelle : présence = téléchargement complet
_SENTINEL = MODEL_DIR / "unet" / "diffusion_pytorch_model.fp16.safetensors"

# Fichiers à exclure :
#   - variantes float32 (x2 en taille, inutiles en fp16)
#   - fichier ComfyUI/A1111 monolithique (6.9 GB)
#   - images d'exemple du repo
_IGNORE = [
    "OpenDalleV1.1.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder_2/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "*.png",
    "*.jpeg",
    "*.jpg",
    "*.webp",
    "*.ckpt",
    "*.pt",
    "*.bin",
]


def is_downloaded():
    return _SENTINEL.exists()


def download():
    if is_downloaded():
        print(f"Modele deja present dans {MODEL_DIR.resolve()}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"  Modele       : {MODEL_ID}")
    print(f"  Destination  : {MODEL_DIR.resolve()}")
    print(f"  Taille aprox : ~7 GB (fp16 uniquement)")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            ignore_patterns=_IGNORE,
        )
    except KeyboardInterrupt:
        print("\nTelechargement interrompu. Relancez pour reprendre.")
        sys.exit(1)

    print("=" * 60)
    print("  Telechargement termine !")
    print(f"  Modele pret dans : {MODEL_DIR.resolve()}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    download()
