"""
scrapper.py
-----------
Data Input layer. Defines SourceDocument (the common shape every sourcegets normalized into) 
and the loaders that populate it: web articles, YouTube transcripts, and local text files.
"""

import re
import requests
from pathlib import Path
from dataclasses import dataclass
from bs4 import BeautifulSoup

from config import DATA_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]


@dataclass
class SourceDocument:
    source_type: str
    origin: str
    raw_text: str


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
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    if not text:
        raise ValueError(f"No article text found at {url} — page structure may need a custom selector.")

    return SourceDocument(source_type="url", origin=url, raw_text=text)


def load_from_youtube(url: str, languages=("en",)) -> SourceDocument:
    """Fetches a YouTube video's transcript. Requires: pip install youtube-transcript-api"""
    from youtube_transcript_api import YouTubeTranscriptApi

    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", url) or re.search(r"youtu\.be/([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    video_id = match.group(1)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
        chunks = transcript.to_raw_data()
    except AttributeError:
        chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))

    text = " ".join(c["text"] for c in chunks)
    if not text.strip():
        raise ValueError(f"No transcript available for {url}")

    return SourceDocument(source_type="youtube", origin=url, raw_text=text)


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


def load_source(source: str) -> SourceDocument:
    """
    Dispatcher: detects whether `source` is a YouTube link or a regular URL,
    and routes it to the correct loader. The UI calls this, not load_from_url()
    directly, so YouTube links work.
    """
    lowered = source.lower().strip()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return load_from_youtube(source)
    elif lowered.startswith("http://") or lowered.startswith("https://"):
        return load_from_url(source)
    else:
        raise ValueError(f"Unrecognized source: {source!r} (expected a web URL or YouTube link)")