"""Project paths and settings shared by every module.

Import from here instead of hard-coding paths, e.g.
    from skillpath.config import PROCESSED_DIR
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded datasets (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned tables, e.g. jobs.parquet
CHROMA_DIR = DATA_DIR / "chroma"        # vector store
UPLOADS_DIR = DATA_DIR / "uploads"      # user resumes, stored locally only
DB_PATH = DATA_DIR / "skillpath.db"     # SQLite: profiles, feedback, sessions
LOG_DIR = ROOT_DIR / "logs"             # agent tool-call logs

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
