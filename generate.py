"""CLI : python generate.py "votre prompt" [--steps 30] [--seed 42]"""
import argparse
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
from datetime import datetime
from download_model import download, MODEL_DIR, MODEL_ID

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generateur d'images OpenDalle V1.1 (offline)")
    parser.add_argument("prompt", help="Texte decrivant l'image a generer")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'etapes (defaut: 30)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (defaut: 7.5)")
    parser.add_argument("--width", type=int, default=1024, help="Largeur en pixels (defaut: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Hauteur en pixels (defaut: 1024)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed pour la reproductibilite (-1 = aleatoire)")
    args = parser.parse_args()

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        download()

    print("Chargement du modele en VRAM...")
    source = str(MODEL_DIR)
    pipe = AutoPipelineForText2Image.from_pretrained(
        source,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed >= 0 else None

    step_count = [0]

    def on_step(pipe, step, timestep, kwargs):
        step_count[0] = step + 1
        bar = "#" * (step_count[0] * 30 // args.steps)
        empty = "-" * (30 - len(bar))
        print(f"\r  [{bar}{empty}] {step_count[0]}/{args.steps}", end="", flush=True)
        return kwargs

    print(f"Generation : {args.prompt!r}")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative or None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
        callback_on_step_end=on_step,
    ).images[0]

    print()
    out = OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(out)
    print(f"Image sauvegardee : {out}")


if __name__ == "__main__":
    main()
