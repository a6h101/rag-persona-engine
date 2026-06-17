"""
build_index.py

Runs the full offline pipeline once:
  1. Load all messages from the CSV
  2. Topic-segment + fixed-100-segment the full stream
  3. Summarize every segment (local Ollama)
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
raising/removing the cap. Change MAX_MESSAGES below as needed.
"""

import time
from data_loader import load_messages
from topic_segmenter import TopicSegmenter
from checkpoint_summarizer import summarize_all, save_summaries
from rag_store import RagStore
from persona_extractor import extract_persona, save_persona

CSV_PATH = "data/conversations.csv"
MAX_MESSAGES = 2000  # cap for demo speed; set to None to process the full file
TARGET_PERSONA_SPEAKER = "User 1"  # which role label to build the persona for
TARGET_PERSONA_ROWS = None  # None = pool across all rows included in MAX_MESSAGES


def main():
    t0 = time.time()
    print("Loading messages...")
    messages = load_messages(CSV_PATH)
    if MAX_MESSAGES:
        messages = messages[:MAX_MESSAGES]
    print(f"  {len(messages)} messages loaded (capped={MAX_MESSAGES is not None})")

    print("Segmenting by topic...")
    segmenter = TopicSegmenter()
    topic_segments = segmenter.segment_by_topic(messages)
    print(f"  {len(topic_segments)} topic segments found")

    print("Segmenting fixed-100...")
    fixed_segments = segmenter.segment_fixed_100(messages)
    print(f"  {len(fixed_segments)} fixed-100 segments found")

    print("Summarizing topic segments (local Ollama)...")
    topic_summaries = summarize_all(topic_segments)
    save_summaries(topic_summaries, "outputs/topic_summaries.json")

    print("Summarizing fixed-100 segments (local Ollama)...")
    fixed_summaries = summarize_all(fixed_segments)
    save_summaries(fixed_summaries, "outputs/fixed100_summaries.json")

    print("Building RAG index (ChromaDB)...")
    store = RagStore()
    store.ingest_summaries(topic_summaries + fixed_summaries)
    store.ingest_message_chunks(messages)
    print("  Index built.")

    print("Extracting persona...")
    if TARGET_PERSONA_ROWS is not None:
        speaker_msgs = [
            m for m in messages
            if m.row_idx in TARGET_PERSONA_ROWS and m.speaker == TARGET_PERSONA_SPEAKER
        ]
    else:
        speaker_msgs = [m for m in messages if m.speaker == TARGET_PERSONA_SPEAKER]

    persona = extract_persona(speaker_msgs, speaker_label=TARGET_PERSONA_SPEAKER, use_llm_labels=True)
    save_persona(persona, "outputs/persona.json")
    print(f"  Persona saved with {len(persona.evidence_log)} evidence items.")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Now run: streamlit run chatbot.py")


if __name__ == "__main__":
    main()
