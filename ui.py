"""
ui.py
------
Gradio Blocks interface. Wires the pipeline (scrapper -> content_processor
-> audio_generator) to a button click.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from scrapper import load_source, save_to_file
from content_processor import condense_source, write_episode_script, generate_episode_metadata
from audio_generator import text_to_speech
from config import HOST_VOICE

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="yellow",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
)

custom_css = """
#header-banner {
    background: linear-gradient(135deg, #c2540a 0%, #e08a1f 100%);
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 16px;
}
#header-banner h1 { font-size: 2.2em; margin: 0; }
#header-banner p { color: #fff3e0; margin-top: 4px; }
.generate-btn { font-size: 1.1em !important; }
"""


def generate_podcast(sources_text, voice, progress=gr.Progress()):
    log_lines = []
    sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
    if not sources:
        raise gr.Error("Please enter at least one source URL.")

    progress(0.0, desc="Loading sources...")
    docs = []
    for src in sources:
        try:
            doc = load_source(src)
            save_to_file(doc)
            docs.append(doc)
            log_lines.append(f"✓ Loaded: {src} [{len(doc.raw_text.split())} words]")
        except Exception as e:
            log_lines.append(f"✗ Failed: {src} — {e}")

    if not docs:
        raise gr.Error("None of the sources could be loaded. Check the log.")

    progress(0.3, desc="Generating title, topic & condensing sources...")
    combined_text = "\n\n".join(doc.raw_text for doc in docs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        meta_future = executor.submit(generate_episode_metadata, combined_text)
        condensed_future = executor.submit(
            lambda: {doc.origin: condense_source(doc) for doc in docs}
        )
        try:
            meta = meta_future.result()
            episode_title, topic = meta.title, meta.topic
        except Exception:
            episode_title, topic = "The Joy of Learning: Episode", "general learning topics"
        condensed_sources = condensed_future.result()

    progress(0.5, desc="Writing script...")
    try:
        script = write_episode_script(
            condensed_sources, episode_title, topic,
            audience="general listeners", target_words=500,
        )
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Script generation failed: {e}")

    progress(0.75, desc="Generating audio...")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in episode_title)
    try:
        audio_path = text_to_speech(script, output_file=f"output/{safe_title or 'episode'}.mp3", voice=HOST_VOICE)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Audio generation failed: {e}")

    progress(1.0, desc="Done!")
    return episode_title, script, audio_path, "\n".join(log_lines)


def build_demo() -> gr.Blocks:
    
    with gr.Blocks(title="The Joy of Learning — Podcast Studio", theme=theme, css=custom_css) as demo:
        gr.HTML(
            """
            <div id="header-banner">
                <h1>The Joy of Learning</h1>
                <p>Turn articles & videos into a warm, witty podcast episode of The Joy of Learning podcast.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                sources_input = gr.Textbox(
                    label=" Sources (one per line — article or YouTube URLs)",
                    lines=8,
                    placeholder="https://example.com/article\nhttps://youtube.com/watch?v=...",
                )
                generate_btn = gr.Button(" GENERATE PODCAST", variant="primary", elem_classes="generate-btn")

            with gr.Column(scale=1):
                title_output = gr.Textbox(label=" Suggested Episode Title", interactive=False)
                topic_output = gr.Textbox(label=" Episode Topics", interactive=False)
                audio_output = gr.Audio(label=" Episode Audio", type="filepath")
                script_output = gr.Textbox(label="Episode Script", lines=12)
                log_output = gr.Textbox(label="Processing Log", lines=6)
        generate_btn.click(
            fn=generate_podcast,
            inputs=[sources_input],
            outputs=[title_output, topic_output, script_output, audio_output, log_output],
    )

    return demo