# The Joy of Learning 🎙️

An AI-generated podcast studio that turns articles and YouTube videos into short, witty audio episodes about the science of learning — written and voiced entirely by AI, hosted by Clare Hayden.

**Authors:** Akansha Verma & Louise Plessis
**Program:** Ironhack — AI Consulting & Integration Bootcamp, Project 1

---

## What it does

Feed the app one or more article or YouTube links. It scrapes or transcribes the content, condenses it, writes a single cohesive podcast script in Clare's voice, generates the audio, and hands you back a ready-to-publish episode — title, topic, script, and audio file — through a simple web interface.

No manual scripting. No manual editing. No recording booth.

## Features

- **Multi-source input** — accepts web articles and YouTube video links, one per line
- **Automatic transcription** — pulls YouTube transcripts directly, no separate download step
- **AI-condensed research** — each source is summarized individually before being woven together (map-reduce pattern)
- **Auto-generated metadata** — episode title and topic are derived from the source content, not hardcoded
- **Consistent host persona** — every episode opens the same way, voiced by "Clare Hayden"
- **Parallelized processing** — metadata generation and source condensing run concurrently; long scripts are chunked and synthesized to audio in parallel
- **Simple web UI** — built with Gradio, no command-line usage required to generate an episode

## How it works

1. **Sources** — you paste article and/or YouTube URLs into the app
2. **Loader** — each source is scraped (web) or transcribed (YouTube) into raw text
3. **Condense** — an LLM call summarizes each source into a short brief
4. **Script writer** — an LLM weaves all briefs into one continuous episode script, in Clare's voice
5. **Text-to-speech** — the script is chunked (to respect API input limits) and synthesized to audio, run in parallel, then stitched into a single file
6. **UI output** — the generated title, topic, script, and audio file are displayed back to you

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3 |
| LLM (text) | OpenAI `gpt-4o-mini` |
| LLM (structured output) | Pydantic + OpenAI structured parsing |
| Text-to-speech | OpenAI `tts-1-hd` |
| Web scraping | `requests` + `BeautifulSoup` |
| YouTube transcripts | `youtube-transcript-api` |
| Audio stitching | `pydub` (+ ffmpeg) |
| UI | Gradio |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` |

## Project structure

```
Project-1-Simple-Podcast/
├── input_data/                  # Cached scraped/transcribed source text
├── output                       # saved episode mp3 files
├── Main Notebook.ipynb          # Primary development notebook (full pipeline)
├── scrapper.py                  # Source loading (URL + YouTube)
├── content_processor.py         # Condensing & script-writing logic
├── audio_generator.py           # Text-to-speech + audio stitching
├── config.py                    # Constants (models, voice, paths)
├── app.py                       # main() to call
├── ui.py                        # Gradio interface
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/lou-PL-dev/Project-1-Simple-Podcast.git
cd Project-1-Simple-Podcast
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
```

## Usage

Run the app:

```bash
python app.py
```

Then open the local Gradio URL printed in the terminal, paste in one or more source URLs (articles or YouTube links), and click **Generate podcast**. The episode title, topic, script, and audio will appear once generation completes (usually under a minute).

## Roadmap

The pipeline is designed to be fully automatable end-to-end:

- [ ] **Automate sourcing** — scheduled job (e.g. GitHub Actions) to pull new articles/videos automatically
- [ ] **Auto-publish via RSS** — add finished episodes straight to a hosted RSS feed
- [ ] **Spotify distribution** — connect the RSS feed to Spotify for Podcasters for automatic ingestion
- [ ] **Release cadence** — publish a new episode every 2 days, hands-off
- [ ] **Monetization** — sponsorships and listener support once the audience grows

## License

Educational project — built as part of the Ironhack AI Consulting & Integration Bootcamp.
