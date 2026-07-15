"""
app.py
-------
Entry point. Run with: python app.py
"""

from config import check_api_key
from ui import build_demo

if __name__ == "__main__":
    check_api_key()
    demo = build_demo()
    demo.launch()