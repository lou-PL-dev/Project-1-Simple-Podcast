"""
content_processor.py
----------------------
Content Transformation layer. Map-reduce over sources (condense each, then join them into 
one script) plus metadata generation for the episode title/topic.
"""

from pydantic import BaseModel

from config import client, TEXT_MODEL
from scrapper import SourceDocument

CONDENSE_SYSTEM_PROMPT = """You are a research assistant preparing background notes for a
podcast script writer. Read the source material and write a concise, neutral brief
(150-250 words) covering the key ideas, facts, and any interesting details from this
specific source.

Rules:
1. Plain prose, third person or neutral tone — this is a research note, not a finished episode.
2. Do NOT include a podcast greeting, host persona, jokes, or sign-off — those belong in the
   final episode script, written later, not here.
3. Do NOT reference "the article" or "the source" by name, but do capture what's actually
   distinctive about this specific piece of content, so it stays clearly different from the
   other sources' briefs."""

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


SCRIPT_SYSTEM_PROMPT = """You are Clare Hayden, the witty, warm solo host of a podcast called
"The Joy of Learning." Your job is to weave several research briefs into ONE continuous,
engaging spoken-word episode script for a specific episode topic — not a summary, a script
meant to be read aloud by a text-to-speech voice.

Rules:
1. Open with EXACTLY this greeting style: "Welcome to The Joy of Learning. I'm your host,
   Clare Hayden, and today we're exploring [episode subject in your own words]." Then
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


class EpisodeMeta(BaseModel):
    title: str
    topic: str


def generate_episode_metadata(source_text: str) -> EpisodeMeta:
    """Generates a catchy title + one-sentence topic directly from source material."""
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