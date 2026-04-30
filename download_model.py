"""
Télécharge OpenDalle V1.1 dans le dossier models/ local.
Exécuter une seule fois avant de lancer app.py.
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

MODEL_ID = "dataautogpt3/OpenDalleV1.1"
MODEL_DIR = Path("models/OpenDalleV1.1")


def download():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"Modele deja present dans {MODEL_DIR.resolve()}")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"  Modele       : {MODEL_ID}")
    print(f"  Destination  : {MODEL_DIR.resolve()}")
    print(f"  Taille aprox : ~7 GB")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            ignore_patterns=["*.ckpt", "*.pt"],
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
