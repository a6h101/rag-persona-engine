"""
checkpoint_summarizer.py

Summarizes each Segment (from topic_segmenter.py) into a short text
summary using Groq API (llama3-8b-8192). Set GROQ_API_KEY env var
before running. Falls back to extractive summary if Groq is unreachable.

Each summary is stored alongside its segment metadata (start/end idx,
kind, row range) so retrieval can cite which messages it came from.
"""

import json
import requests
from dataclasses import dataclass, asdict
from typing import List

from topic_segmenter import Segment

# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3.2"
# changing from llama to groq for cloud deployment
import os

import streamlit as st

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@dataclass
class CheckpointSummary:
    checkpoint_id: str
    kind: str  # "topic" or "fixed100"
    start_idx: int
    end_idx: int
    row_range: tuple  # (min_row_idx, max_row_idx) covered by this segment
    summary: str
    message_count: int


def _format_segment_text(segment: Segment) -> str:
    lines = [f"{m.speaker}: {m.text}" for m in segment.messages]
    return "\n".join(lines)



def _call_ollama(prompt: str) -> str:
    
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30,
        allow_redirects=False
    )
    print("Status:", resp.status_code)
    print("Response:", resp.text)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def summarize_segment(segment: Segment, idx_in_kind: int) -> CheckpointSummary:
    text = _format_segment_text(segment)
    row_indices = [m.row_idx for m in segment.messages]
    row_range = (min(row_indices), max(row_indices))

    label = "topic segment" if segment.kind == "topic" else "100-message checkpoint block"
    prompt = (
        f"You are summarizing a {label} from a conversation log. "
        f"Write a concise 2-3 sentence summary capturing what was discussed, "
        f"any key facts, preferences, or events mentioned. "
        f"Do not repeat the raw dialogue, synthesize it.\n\n"
        f"--- CONVERSATION SEGMENT ---\n{text}\n--- END SEGMENT ---\n\n"
        f"Summary:"
    )

    try:
        summary_text = _call_ollama(prompt)
    # ADD:
    except requests.exceptions.RequestException as e:
        summary_text = (
            "[Groq unavailable — extractive fallback] "
            + " / ".join(m.text for m in segment.messages[:3])
        )

    checkpoint_id = f"{segment.kind}_{idx_in_kind:05d}"
    return CheckpointSummary(
        checkpoint_id=checkpoint_id,
        kind=segment.kind,
        start_idx=segment.start_idx,
        end_idx=segment.end_idx,
        row_range=row_range,
        summary=summary_text,
        message_count=len(segment.messages),
    )


def summarize_all(segments: List[Segment]) -> List[CheckpointSummary]:
    summaries = []
    for i, seg in enumerate(segments):
        summaries.append(summarize_segment(seg, i))
        print(f"  Summarized {seg.kind} segment {i+1}/{len(segments)} "
              f"[{seg.start_idx}:{seg.end_idx}]")
    return summaries


def save_summaries(summaries: List[CheckpointSummary], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)


if __name__ == "__main__":
    from data_loader import load_messages
    from topic_segmenter import TopicSegmenter

    msgs = load_messages("data/conversations.csv")
    sample = msgs[:300]

    seg = TopicSegmenter()
    topic_segments = seg.segment_by_topic(sample)
    fixed_segments = seg.segment_fixed_100(sample)

    print("Summarizing topic segments...")
    topic_summaries = summarize_all(topic_segments)
    print("Summarizing fixed-100 segments...")
    fixed_summaries = summarize_all(fixed_segments)

    save_summaries(topic_summaries, "outputs/topic_summaries_sample.json")
    save_summaries(fixed_summaries, "outputs/fixed100_summaries_sample.json")

    print("\nSample topic summary:")
    print(json.dumps(asdict(topic_summaries[0]), indent=2))
