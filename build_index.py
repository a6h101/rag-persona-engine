"""
build_index.py

Runs the full offline pipeline once:
  1. Load all messages from the CSV
  2. Topic-segment + fixed-100-segment the full stream
  3. Summarize every segment (local Ollama, with extractive fallback)
  4. Ingest summaries + raw message chunks into ChromaDB
  5. Extract a persona profile and save it as JSON

Run this once before launching chatbot.py:
    python build_index.py

NOTE ON SCALE: the full dataset is ~191k messages. Topic segmentation
embeds every message (fast, seconds on CPU with MiniLM), but Ollama
summarization of every single segment can take a while if you run
it over the ENTIRE corpus (~1900+ fixed-100 checkpoints alone). For
the assessment demo, this script defaults to a configurable MAX_MESSAGES
cap so you can demo on a meaningful, fast-to-process slice and document
in your README that the same pipeline scales to the full file by
raising/removing the cap. Override via the MAX_MESSAGES env var
(set it to "None" to process the full file).


NOTE ON LLM: summarization uses Groq API (llama3-8b-8192, free tier).
Set GROQ_API_KEY env var before running. If Groq is unreachable,
falls back to extractive summary instead of crashing.
"""

import os
import time
import logging

from data_loader import load_messages
from topic_segmenter import TopicSegmenter
from checkpoint_summarizer import summarize_all, save_summaries
from rag_store import RagStore
from persona_extractor import extract_persona, save_persona

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CSV_PATH = os.getenv("CSV_PATH", "data/conversations.csv")

_max_messages_raw = os.getenv("MAX_MESSAGES", "10000")
MAX_MESSAGES = None if _max_messages_raw.strip().lower() == "none" else int(_max_messages_raw)

TARGET_PERSONA_SPEAKER = os.getenv("TARGET_PERSONA_SPEAKER", "User 1")
TARGET_PERSONA_ROWS = None  # None = pool across all rows included in MAX_MESSAGES

OUTPUT_DIR = "outputs"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = os.path.dirname(CSV_PATH)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)


def main():
    t0 = time.time()
    ensure_dirs()

    logger.info("Loading messages...")
    try:
        messages = load_messages(CSV_PATH)
    except FileNotFoundError as e:
        logger.error(f"CSV not found at '{CSV_PATH}'. Set CSV_PATH env var or place the file there.")
        raise SystemExit(1) from e

    if MAX_MESSAGES:
        messages = messages[:MAX_MESSAGES]
    logger.info(f"{len(messages)} messages loaded (capped={MAX_MESSAGES is not None})")

    if not messages:
        logger.error("No messages loaded. Check CSV_PATH and file contents.")
        raise SystemExit(1)

    logger.info("Segmenting by topic...")
    try:
        segmenter = TopicSegmenter()
        topic_segments = segmenter.segment_by_topic(messages)
        logger.info(f"{len(topic_segments)} topic segments found")
    except Exception:
        logger.exception("Topic segmentation failed.")
        raise SystemExit(1)

    logger.info("Segmenting fixed-100...")
    try:
        fixed_segments = segmenter.segment_fixed_100(messages)
        logger.info(f"{len(fixed_segments)} fixed-100 segments found")
    except Exception:
        logger.exception("Fixed-100 segmentation failed.")
        raise SystemExit(1)

    logger.info("Summarizing topic segments (local Ollama, with fallback)...")
    try:
        topic_summaries = summarize_all(topic_segments)
    except Exception:
        logger.exception("Topic summarization failed even with fallback. Aborting.")
        raise SystemExit(1)
    save_summaries(topic_summaries, os.path.join(OUTPUT_DIR, "topic_summaries.json"))

    logger.info("Summarizing fixed-100 segments (local Ollama, with fallback)...")
    try:
        fixed_summaries = summarize_all(fixed_segments)
    except Exception:
        logger.exception("Fixed-100 summarization failed even with fallback. Aborting.")
        raise SystemExit(1)
    save_summaries(fixed_summaries, os.path.join(OUTPUT_DIR, "fixed100_summaries.json"))

    logger.info("Building RAG index (ChromaDB)...")
    try:
        store = RagStore()
        store.ingest_summaries(topic_summaries + fixed_summaries)
        store.ingest_message_chunks(messages)
        logger.info("Index built.")
    except Exception:
        logger.exception("RAG index build failed.")
        raise SystemExit(1)

    logger.info("Extracting persona...")
    try:
        if TARGET_PERSONA_ROWS is not None:
            speaker_msgs = [
                m for m in messages
                if m.row_idx in TARGET_PERSONA_ROWS and m.speaker == TARGET_PERSONA_SPEAKER
            ]
        else:
            speaker_msgs = [m for m in messages if m.speaker == TARGET_PERSONA_SPEAKER]

        if not speaker_msgs:
            logger.warning(
                f"No messages found for speaker '{TARGET_PERSONA_SPEAKER}'. "
                "Persona will be empty. Check TARGET_PERSONA_SPEAKER and MAX_MESSAGES."
            )

        persona = extract_persona(
            speaker_msgs, speaker_label=TARGET_PERSONA_SPEAKER, use_llm_labels=True
        )
        save_persona(persona, os.path.join(OUTPUT_DIR, "persona.json"))
        logger.info(f"Persona saved with {len(persona.evidence_log)} evidence items.")
    except Exception:
        logger.exception("Persona extraction failed.")
        raise SystemExit(1)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. Now run: streamlit run chatbot.py")


if __name__ == "__main__":
    main()