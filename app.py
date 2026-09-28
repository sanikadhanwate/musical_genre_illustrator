"""
PHASE 5: Full pipeline, connected.
Genre classification is always LOCAL. Image generation can be toggled
between LOCAL (tiny-sd) and REMOTE (LLM prompt-writer + Qwen-Image API).
Run: python app.py
"""

import os
import torch
from datetime import datetime

import gradio as gr
import spaces
from transformers import pipeline
from huggingface_hub import InferenceClient
from diffusers import DiffusionPipeline

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GENRE_MODEL_ID = "dima806/music_genres_classification"
LOCAL_IMAGE_MODEL_ID = "segmind/tiny-sd"
TEXT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REMOTE_IMAGE_MODEL_ID = "Qwen/Qwen-Image"

HF_TOKEN = os.environ.get("HF_TOKEN")

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FALLBACK_PROMPTS = {
    "rock": "a gritty, high-energy illustration with electric guitars, sparks, bold red and black tones",
    "pop": "a bright, glossy, colorful pop-art style illustration full of energy and glitter",
    "jazz": "a moody, smoky illustration of a jazz club with warm amber lighting and saxophones",
    "classical": "an elegant illustration of an orchestra hall with soft golden light and violins",
    "hiphop": "an urban street-art style illustration with bold graffiti colors and city skyline",
    "country": "a warm, rustic illustration of open fields, a guitar, and a sunset",
    "disco": "a vibrant retro illustration with disco balls, neon lights, and 70s colors",
    "metal": "a dark, intense illustration with jagged shapes, fire, and heavy shadows",
    "reggae": "a relaxed, sun-drenched illustration in green-yellow-red tones with palm trees",
    "blues": "a melancholic blue-toned illustration of a lone guitarist under a streetlamp",
}

# ---------------------------------------------------------------------------
# LOAD MODELS ONCE AT STARTUP
# ---------------------------------------------------------------------------
print("Loading local genre classifier...")
classifier = pipeline("audio-classification", model=GENRE_MODEL_ID)

print("Loading local image generator (tiny-sd)...")
local_image_pipe = DiffusionPipeline.from_pretrained(
    LOCAL_IMAGE_MODEL_ID, torch_dtype=torch.float32
)
# NOTE: do NOT call .to("cuda") here. Under ZeroGPU, CUDA can only be
# touched inside a function decorated with @spaces.GPU (see generate_image_local).

def get_client(hf_token: gr.OAuthToken = None):
    token = getattr(hf_token, "token", None)

    if not token:
        print("Login required.")
        return None

    client = InferenceClient(token=token)
    print("API model ready.")
    return client

# ---------------------------------------------------------------------------
# STEP 1: Genre classification (LOCAL)
# ---------------------------------------------------------------------------
@spaces.GPU
def classify_audio(audio_file):
    if audio_file is None:
        return "unknown", 0.0
    result = classifier(audio_file)
    genre = result[0]["label"]
    confidence = result[0]["score"]
    return genre, confidence


# ---------------------------------------------------------------------------
# STEP 2: Genre -> creative prompt (REMOTE LLM)
# ---------------------------------------------------------------------------
def create_visual_prompt(genre, hf_token):
    instruction = f"""
    The uploaded music has been classified as {genre}.

    Create a detailed artistic prompt for an image generator.
    Capture the mood, atmosphere, instruments, colors,
    energy, and visual identity associated with {genre}.

    Return only the image generation prompt, nothing else.
    """
    try:
        client = get_client(hf_token)

        if client is None:
            raise Exception("Please log in with Hugging Face.")
        response = client.chat_completion(
            model=TEXT_MODEL_ID,
            messages=[{"role": "user", "content": instruction}],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Remote LLM failed, using fallback prompt: {e}")
        return FALLBACK_PROMPTS.get(
            genre.lower(), f"a colorful abstract illustration representing {genre} music"
        )


# ---------------------------------------------------------------------------
# STEP 3: Prompt -> image, LOCAL or REMOTE
# ---------------------------------------------------------------------------
@spaces.GPU
def generate_image_local(prompt):
    local_image_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = local_image_pipe(prompt, num_inference_steps=15).images[0]
    return image


def generate_image_remote(prompt, hf_token):
    try:
        client = get_client(hf_token)

        if client is None:
            raise Exception("Please log in with Hugging Face.")
        return client.text_to_image(prompt, model=REMOTE_IMAGE_MODEL_ID)
    except Exception as e:
        print(f"[ERROR] Remote image generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------
def analyze_music(audio_file, use_local_image_gen, hf_token: gr.OAuthToken=None):
    if audio_file is None:
        return "No file uploaded", "N/A", "N/A", None, None

    # STEP 1
    genre, confidence = classify_audio(audio_file)

    # STEP 2: only needed for the remote path; local path uses a simple template
    if use_local_image_gen:
        visual_prompt = FALLBACK_PROMPTS.get(
            genre.lower(), f"a colorful abstract illustration representing {genre} music"
        )
        image = generate_image_local(visual_prompt)
    else:
        visual_prompt = create_visual_prompt(genre, hf_token)
        image = generate_image_remote(visual_prompt, hf_token)

    saved_path = None
    if image is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join(OUTPUT_DIR, f"{genre}_{timestamp}.png")
        image.save(saved_path)

    return genre, f"{confidence:.2f}", visual_prompt, image, saved_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Music-to-Art Generator") as demo:
    gr.Markdown("# 🎵 Music-to-Art Generator")
    gr.Markdown(
        "Upload audio. A local model detects the genre. Toggle below to "
        "generate the artwork **locally (tiny-sd)** or **remotely (LLM + Qwen-Image API)**."
    )

    audio_input = gr.Audio(type="filepath", label="Upload Audio")
    gr.LoginButton()
    use_local_toggle = gr.Checkbox(label="Use Local Model for image generation", value=False)
    analyze_btn = gr.Button("Analyze Music", variant="primary")

    with gr.Row():
        genre_output = gr.Textbox(label="Genre")
        confidence_output = gr.Textbox(label="Confidence")

    prompt_output = gr.Textbox(label="AI Interpretation", lines=3)
    image_output = gr.Image(label="Generated Artwork")
    file_output = gr.File(label="Saved image file")

    analyze_btn.click(
        fn=analyze_music,
        inputs=[audio_input, use_local_toggle],
        outputs=[genre_output, confidence_output, prompt_output, image_output, file_output],
    )

if __name__ == "__main__":
    demo.launch()
