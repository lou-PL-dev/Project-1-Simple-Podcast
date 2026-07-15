"""
config.py
----------
Shared configuration file 
"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("input_data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TEXT_MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1-hd"
HOST_VOICE = "nova"

def check_api_key() -> bool:
    loaded = bool(os.getenv("OPENAI_API_KEY"))
    print("Key loaded:", loaded)
    return loaded