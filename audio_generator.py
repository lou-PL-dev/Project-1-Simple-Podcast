"""
audio_generator.py
--------------------
Audio Generation layer. Splits a script into TTS-safe chunks, synthesizes
them concurrently, and stitches the results into one mp3.
"""

import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from config import client, TTS_MODEL

MAX_CHARS = 3800  # safety margin under the ~4096-char TTS input limit

def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list:
    import re
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