"""
PHASE 5: Full pipeline, connected.
Genre classification is always LOCAL. Image generation can be toggled
between LOCAL (tiny-sd) and REMOTE (LLM prompt-writer + Qwen-Image API).
Run: python app.py
"""

import spaces
import gradio as gr

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
