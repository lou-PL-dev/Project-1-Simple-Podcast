# # Project 1: Create a simple Podcast - The Joy of Learning
# Authors: Akansha Verma & Louise Plessis
#  

## 0. Setup
# Initializes the environment, loads credentials, and sets project-wide constants used later.

import os
import re
import io
import requests
import gradio as gr
import traceback
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

DATA_DIR = Path("input_data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- FIX: Separate Text and Audio Models ---
TEXT_MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1"

# ## Scrapper
# Defines `SourceDocument`, the common structure every source gets normalized into, plus the 
# functions that populate it: `load_from_url()` scrapes and cleans article text from a 
# webpage, `load_from_text_file()/load_all_from_folder()` read local `.txt` `files, and 
# save_to_file()` caches scraped content to input_data/ so sources aren't re-scraped on 
# every run.
# Data structure & constants (run once)
@dataclass
class SourceDocument:
    source_type: str
    origin: str
    raw_text: str


HEADERS = {
    # Some sites block requests with no User-Agent; a normal browser UA avoids that.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Tags that are almost never part of the actual article content.
NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]

# Functions (run once)
def _slugify(url: str) -> str:
    """Turns a URL into a safe filename, e.g. mindfulambition-net-beginners-mind.txt"""
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return slug[:100]


def save_to_file(doc: SourceDocument, folder: Path = DATA_DIR) -> Path:
    """Saves a SourceDocument's raw_text to 'input_data/<slug>.txt' and returns the path."""
    filename = f"{_slugify(doc.origin)}.txt"
    out_path = folder / filename
    out_path.write_text(doc.raw_text, encoding="utf-8")
    return out_path


def load_from_url(url: str, timeout: int = 15) -> SourceDocument:
    """Downloads the page at `url` and extracts readable article text."""
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    container = (
        soup.find("article")
        or soup.find("div", class_=lambda c: c and "entry-content" in c)
        or soup.find("div", class_=lambda c: c and "post-content" in c)
        or soup.find("main")
    )
    paragraphs = container.find_all("p") if container else soup.find_all("p")

    text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
    )

    if not text:
        raise ValueError(f"No article text found at {url} — page structure may need a custom selector.")

    return SourceDocument(source_type="url", origin=url, raw_text=text)

def load_from_text_file(path: str) -> SourceDocument:
    text = Path(path).read_text(encoding="utf-8")
    return SourceDocument(source_type="text", origin=str(path), raw_text=text)

def load_all_from_folder(folder: str = "input_data", pattern: str = "*.txt") -> list:
    """Reads every .txt file in `folder` and returns one SourceDocument per file."""
    folder_path = Path(folder)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {folder_path.resolve()}")
    return [load_from_text_file(str(f)) for f in files]


## Process sources ( Content Transformation)
# Implements the map-reduce transformation stage: each source is condensed into a short 
# brief (map), then all briefs are joined into one unified episode script (reduce).

CONDENSE_SYSTEM_PROMPT="""You are the witty, warm solo host of a podcast called 'The Joy of Learning.' 
Your job is to turn source material into an engaging spoken-word script — not a summary, a script meant 
to be read aloud by a text-to-speech voice.
Rules:
1. Open with exactly this greeting style: 'Welcome to The Joy of Learning, the podcast where we 
explore [general subject in your own words].' Then transition naturally into the episode.
2. Explain the core concept in your own words — do not summarize the source paragraph by paragraph, and 
do not reference the article, its author, or where the content came from. Write as if this is your own 
idea you're excited to share.
3. Be genuinely funny: use a playful analogy, a light joke, or a self-aware aside at least once. 
Think 'smart friend explaining something cool over coffee,' not 'lecture.'
4. Write only what should be spoken aloud — no headers, no stage directions, no markdown, 
no sound-effect cues.
5. Keep it under 300 words so the TTS output stays a reasonable length.
6. End with a short, warm sign-off that invites the listener to try adopting the mindset themselves.
"""
def condense_source(doc: SourceDocument, max_chars: int = 12000) -> str:
    text = doc.raw_text[:max_chars]
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content

SCRIPT_SYSTEM_PROMPT = """You are the witty, warm solo host of a podcast called
"The Joy of Learning." Your job is to weave several research briefs into ONE
continuous, engaging spoken-word episode script — not a summary, a script meant
to be read aloud by a text-to-speech voice.

Rules:
1. Open with exactly this greeting style: "Welcome to The Joy of Learning, the
   podcast where we explore [general subject in your own words]." Then
   transition naturally into the episode.
2. Weave the ideas from ALL the briefs into a single narrative arc with smooth
   transitions between them — do not treat them as separate segments.
3. Explain concepts in your own words. Never reference "the article," "the
   source," or where the content came from.
4. Be genuinely funny at least once every couple of minutes of runtime.
5. Write only what should be spoken aloud — no headers, no markdown, no
   stage directions.
6. End with a short, warm sign-off.
7. STRICT LENGTH: the script MUST be close to {target_words} words. This is a
   hard constraint, not a suggestion. Do not exceed it by more than 10%."""

def write_episode_script(condensed_sources: dict, episode_title, topic, audience, target_words=500) -> str:
    system_prompt = SCRIPT_SYSTEM_PROMPT.format(target_words=target_words, num_sources=len(condensed_sources))
    briefs_block = "\n\n".join(
        f"--- Brief {i+1} (from {origin}) ---\n{notes}"
        for i, (origin, notes) in enumerate(condensed_sources.items())
    )
    user_prompt = (
        f"Episode title: {episode_title}\n"
        f"Episode topic: {topic}\n"
        f"Target audience: {audience}\n\n"
        f"Here are the research briefs to weave together:\n\n{briefs_block}"
    )
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content


def generate_episode_script(docs, episode_title, topic, audience, target_words=500):
    """Full transformation step: condense every doc, then write the unified script."""
    condensed = {doc.origin: condense_source(doc) for doc in docs}
    return write_episode_script(condensed, episode_title, topic, audience, target_words)


# ## Audio Generation (Text-to-Speech)
# Converts the final episode script into a single playable mp3 file. Handles the fact that 
# OpenAI's TTS endpoint has an input-length limit, by splitting long scripts into chunks, 
# synthesizing them concurrently, and stitching the results back into one audio file.

MAX_CHARS = 3800  # safety margin under the ~4096-char TTS input limit

def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def synthesize_chunk(text: str, voice: str = "alloy", model: str = TTS_MODEL) -> bytes:
    response = client.audio.speech.create(model=model, voice=voice, input=text)
    return response.content


def _concatenate_mp3(audio_chunks: list) -> bytes:
    """Uses pydub+ffmpeg for a clean join if available, else raw byte concat."""
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for chunk_bytes in audio_chunks:
            combined += AudioSegment.from_file(io.BytesIO(chunk_bytes), format="mp3")
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3")
        return buffer.getvalue()
    except Exception as e:
        print(f"pydub/ffmpeg unavailable ({e}); falling back to raw byte concatenation.")
        return b"".join(audio_chunks)


def text_to_speech(text: str, output_file: str = "output/episode.mp3", voice: str = "alloy") -> str:
    chunks = chunk_text(text)
    print(f"Synthesizing {len(chunks)} chunk(s)...")
    with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as executor:
        audio_bytes_list = list(executor.map(lambda c: synthesize_chunk(c, voice=voice), chunks))
    combined_audio = _concatenate_mp3(audio_bytes_list)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(combined_audio)
    print(f"Audio saved to {out_path}")
    return str(out_path) 

# Manual test of the pipeline so far
"""
docs = load_all_from_folder("input_data")
print(f"Loaded {len(docs)} source(s):", [d.origin for d in docs])

script = generate_episode_script(
    docs,
    episode_title="The Joy of Learning: Embracing the Curve",
    topic="the learning curve, cognitive science, beginner's mind, and learning in groups",
    audience="adult learners currently going through a steep learning curve",
    target_words=500,
)
print(script)
"""

## Create Meta data (title and topic)
# Automatically generates a catchy episode title and one-sentence topic description directly 
# from the source material, using structured LLM output instead of manually specifying them.

class EpisodeMeta(BaseModel):
    title: str
    topic: str

def generate_episode_metadata(source_text: str) -> EpisodeMeta:
    response = client.beta.chat.completions.parse(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": (
                "You generate metadata for episodes of a podcast called "
                "'The Joy of Learning.' Given source material, return:\n"
                "- title: a catchy, specific episode title (max 8 words), "
                "in the style of 'The Joy of Learning: <hook>'\n"
                "- topic: one sentence describing what the episode covers."
            )},
            {"role": "user", "content": f"Source material:\n\n{source_text[:6000]}"}
        ],
        response_format=EpisodeMeta,
    )
    return response.choices[0].message.parsed

# Generate audio (test run)
"""
audio_path = text_to_speech(script, output_file="output/episode.mp3", voice="alloy")
audio_path
"""

## Create UI Theme
# Defines the visual theme and custom CSS for the Gradio interface

## Create UI

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

# ## Gradio Interface & Launch
# Assembles the full pipeline into an interactive web app — orchestrates loading sources, 
# generating metadata, condensing, script-writing, and audio synthesis behind a single button,
# then defines and launches the UI itself.

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

def generate_podcast(sources_text, voice, progress=gr.Progress()):
    log_lines = []
    sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
    if not sources:
        raise gr.Error("Please enter at least one source URL.")

    progress(0.0, desc="Loading sources...")
    docs = []
    for src in sources:
        try:
            doc = load_from_url(src)
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
        audio_path = text_to_speech(script, output_file=f"output/{safe_title or 'episode'}.mp3", voice=voice)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Audio generation failed: {e}")

    progress(1.0, desc="Done!")
    return episode_title, script, audio_path, "\n".join(log_lines)


with gr.Blocks(title="The Joy of Learning — Podcast Studio", theme=theme, css=custom_css) as demo:
    gr.HTML(
        """
        <div id="header-banner">
            <h1>The Joy of Learning</h1>
            <p>Turn articles & videos into a warm, witty podcast episode — automatically.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            sources_input = gr.Textbox(
                label="📥 Sources (one per line — article or YouTube URLs)",
                lines=8,
                placeholder="https://example.com/article\nhttps://youtube.com/watch?v=...",
            )
            voice_input = gr.Dropdown(VOICES, value="alloy", label="🎤 Voice")
            generate_btn = gr.Button("✨ Generate Podcast", variant="primary", elem_classes="generate-btn")

        with gr.Column(scale=1):
            title_output = gr.Textbox(label="📝 Generated Episode Title", interactive=False)
            audio_output = gr.Audio(label="🔊 Episode Audio", type="filepath")
            script_output = gr.Textbox(label="Episode Script", lines=12)
            log_output = gr.Textbox(label="Pipeline Log", lines=6)
    generate_btn.click(
        fn=generate_podcast,
        inputs=[sources_input, voice_input],
        outputs=[title_output, script_output, audio_output, log_output],
    )
if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    demo.launch()
