"""
url_scraper.py
---------------
Fetches an article URL and extracts its main body text, returning a
SourceDocument (same structure used elsewhere in data_processor.py)
so it can be handed straight to llm_processor.py.

Install:
    pip install requests beautifulsoup4

Usage:
    python url_scraper.py https://mindfulambition.net/beginners-mind/
"""

import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

DATA_DIR = Path("input data")
DATA_DIR.mkdir(exist_ok=True)


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


def _slugify(url: str) -> str:
    """Turns a URL into a safe filename, e.g. mindfulambition-net-beginners-mind.txt"""
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return slug[:100]  # keep filenames reasonable


def save_to_file(doc: SourceDocument, folder: Path = DATA_DIR) -> Path:
    """Saves a SourceDocument's raw_text to data/<slug>.txt and returns the path."""
    filename = f"{_slugify(doc.origin)}.txt"
    out_path = folder / filename
    out_path.write_text(doc.raw_text, encoding="utf-8")
    return out_path


def load_from_url(url: str, timeout: int = 15) -> SourceDocument:
    """
    Downloads the page at `url` and extracts readable article text.

    Strategy:
      1. Try common article containers (<article>, common CMS classes)
         first, since that gives the cleanest extraction.
      2. Fall back to all <p> tags on the page if no container is found.
      3. Strip nav/header/footer/script/style noise either way.
    """
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

    if container:
        paragraphs = container.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
    )

    if not text:
        raise ValueError(f"No article text found at {url} — page structure may need a custom selector.")

    return SourceDocument(source_type="url", origin=url, raw_text=text)


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://mindfulambition.net/beginners-mind/"
    doc = load_from_url(target_url)
    saved_path = save_to_file(doc)

    print(f"Source: {doc.origin}")
    print(f"Extracted {len(doc.raw_text)} characters, {len(doc.raw_text.split())} words")
    print(f"Saved to: {saved_path}")
    print("---")
    print(doc.raw_text[:300], "...")  # preview only

    doc = load_from_url("https://mindfulambition.net/beginners-mind/")
    saved_path = save_to_file(doc)

    print(f"Source: {doc.origin}")
    print(f"Extracted {len(doc.raw_text)} characters, {len(doc.raw_text.split())} words")
    print(f"Saved to: {saved_path}")
    print("---")
    print(doc.raw_text[:300], "...")