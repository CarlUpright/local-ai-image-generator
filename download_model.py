"""
Télécharge OpenDalle V1.1 dans le dossier models/ local.
Exécuter une seule fois avant de lancer app.py.

On télécharge uniquement les variantes fp16 (~7 GB au lieu de 27 GB).
Le chargement se fait ensuite avec variant="fp16" dans from_pretrained().

Pour des téléchargements rapides (50+ MB/s au lieu de ~150 kB/s), créez un
token HuggingFace gratuit sur https://huggingface.co/settings/tokens
puis définissez la variable d'environnement HF_TOKEN avant de lancer ce script.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "dataautogpt3/OpenDalleV1.1"
MODEL_DIR = Path("models/OpenDalleV1.1")

# Fichier sentinelle : sa présence confirme que le téléchargement est complet
_SENTINEL = MODEL_DIR / "unet" / "diffusion_pytorch_model.fp16.safetensors"

# On utilise allow_patterns (liste blanche) plutôt qu'ignore_patterns
# pour garantir qu'aucun fichier non voulu ne soit téléchargé.
# Seules les variantes fp16 + configs + tokenizers sont nécessaires pour diffusers.
_ALLOW = [
    "model_index.json",
    "*.fp16.safetensors",   # poids fp16 de tous les composants (~7 GB)
    "**/*.json",             # configs (unet, vae, text encoders, scheduler...)
    "**/vocab.json",
    "**/merges.txt",
    "**/tokenizer_config.json",
    "**/special_tokens_map.json",
]


def is_downloaded():
    return _SENTINEL.exists()


def _check_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print()
        print("  AVERTISSEMENT : Aucun HF_TOKEN detecte.")
        print("  La vitesse sera limitee (~150 kB/s) par HuggingFace.")
        print("  Pour 50+ MB/s :")
        print("    1. Creez un token sur https://huggingface.co/settings/tokens")
        print("    2. Lancez : set HF_TOKEN=hf_xxxxxxxxxxxx")
        print("    3. Relancez start.bat")
        print()
    return token


def download():
    if is_downloaded():
        print(f"Modele deja present dans {MODEL_DIR.resolve()}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    token = _check_token()

    print("=" * 60)
    print(f"  Modele       : {MODEL_ID}")
    print(f"  Destination  : {MODEL_DIR.resolve()}")
    print(f"  Taille       : ~7 GB (fp16 uniquement, 27 GB ignores)")
    print(f"  Token HF     : {'oui' if token else 'non (lent)'}")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=_ALLOW,
            token=token or None,
        )
    except KeyboardInterrupt:
        print("\nTelechargement interrompu. Relancez pour reprendre (il continuera ou il en etait)")
        sys.exit(1)

    print("=" * 60)
    print("  Telechargement termine !")
    print(f"  Modele pret dans : {MODEL_DIR.resolve()}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    download()
