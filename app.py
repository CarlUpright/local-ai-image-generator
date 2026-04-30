import torch
import gradio as gr
from diffusers import AutoPipelineForText2Image
from pathlib import Path
from datetime import datetime
from download_model import download, MODEL_DIR, MODEL_ID

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print("Modele absent, lancement du telechargement...")
        download()

    print("Chargement du modele en memoire GPU...")
    source = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_ID
    pipe = AutoPipelineForText2Image.from_pretrained(
        source,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=(source != MODEL_ID),
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("Modele pret.")


def model_status():
    if pipe is not None:
        return "Modele charge en VRAM — pret a generer"
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return "Modele telecharge — sera charge au premier clic sur Generer"
    return "Modele non telecharge — lancer download_model.py ou cliquer sur Generer"


def generate(prompt, negative_prompt, steps, guidance_scale, width, height, seed, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Le prompt ne peut pas etre vide.")

    if pipe is None:
        progress(0, desc="Chargement du modele (peut prendre quelques minutes)...")
        load_model()

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    def gradio_callback(step, timestep, latents):
        progress(step / int(steps), desc=f"Etape {step}/{int(steps)}")

    progress(0, desc="Demarrage de la generation...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=guidance_scale,
        width=int(width),
        height=int(height),
        generator=generator,
        callback_on_step_end=None,
        callback=gradio_callback,
        callback_steps=1,
    ).images[0]

    filename = OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(filename)
    progress(1.0, desc="Image generee !")
    return image, str(filename)


with gr.Blocks(title="OpenDalle V1.1 — Local", theme=gr.themes.Default()) as demo:
    gr.Markdown("## OpenDalle V1.1 — Generation d'images locale (offline)")

    status_box = gr.Textbox(
        value=model_status,
        label="Etat du modele",
        interactive=False,
        every=3,
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="a photorealistic portrait of an astronaut on Mars, golden hour, 8k",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt (optionnel)",
                placeholder="blurry, low quality, distorted",
                lines=2,
            )
            with gr.Row():
                steps = gr.Slider(10, 50, value=30, step=1, label="Etapes")
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            with gr.Row():
                width = gr.Dropdown([512, 768, 1024], value=1024, label="Largeur")
                height = gr.Dropdown([512, 768, 1024], value=1024, label="Hauteur")
            seed = gr.Number(value=-1, label="Seed (-1 = aleatoire)", precision=0)
            btn = gr.Button("Generer", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Image generee", type="pil")
            output_path = gr.Textbox(label="Fichier sauvegarde", interactive=False)

    btn.click(
        generate,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output_image, output_path],
    )

if __name__ == "__main__":
    print("Interface disponible sur http://127.0.0.1:7860")
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
