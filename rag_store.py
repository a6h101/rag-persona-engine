"""
rag_store.py

Builds and queries the RAG index using ChromaDB with two collections:

  1. "topic_summaries"  -> one entry per topic checkpoint (and per
     fixed-100 checkpoint), embedding = the summary text.
  2. "message_chunks"   -> one entry per small chunk of raw messages
     (default: 10 messages per chunk, sliding by 5), embedding = the
     raw chunk text. This lets retrieval pull exact original wording,
     not just summarized gist, when a question needs specifics.

RETRIEVAL LOGIC (combining both):
  Given a query, we embed it once and search both collections.
  - Top-k topic/checkpoint summaries give broad context ("what general
    topics relate to this question").
  - Top-k message chunks give grounded, specific evidence (exact lines
    that support an answer).
  Both result sets are concatenated into a single context block, with
  summaries first (orientation) followed by raw chunks (evidence),
  which is then handed to the chatbot's answer-generation step.
"""

import json
from dataclasses import dataclass, asdict
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from data_loader import Message
from checkpoint_summarizer import CheckpointSummary

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 10
CHUNK_STRIDE = 5


@dataclass
class RetrievedItem:
    source: str  # "topic_summary" or "message_chunk"
    text: str
    metadata: dict
    distance: float


class RagStore:
    def __init__(self, persist_dir: str = "storage/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.summary_collection = self.client.get_or_create_collection("topic_summaries")
        self.chunk_collection = self.client.get_or_create_collection("message_chunks")

    # ---------- INGESTION ----------

    def ingest_summaries(self, summaries: List[CheckpointSummary]):
        if not summaries:
            return
        ids = [s.checkpoint_id for s in summaries]
        docs = [s.summary for s in summaries]
        metadatas = [
            {
                "kind": s.kind,
                "start_idx": s.start_idx,
                "end_idx": s.end_idx,
                "row_min": s.row_range[0],
                "row_max": s.row_range[1],
                "message_count": s.message_count,
            }
            for s in summaries
        ]
        embeddings = self.embedder.encode(docs, show_progress_bar=False).tolist()
        self.summary_collection.upsert(
            ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings
        )

    def ingest_message_chunks(self, messages: List[Message]):
        if not messages:
            return
        chunk_ids, chunk_docs, chunk_metas = [], [], []

        for start in range(0, len(messages), CHUNK_STRIDE):
            end = min(start + CHUNK_SIZE, len(messages))
            if end - start < 2:
                continue
            chunk_msgs = messages[start:end]
            text = "\n".join(f"{m.speaker}: {m.text}" for m in chunk_msgs)
            chunk_id = f"chunk_{start:06d}_{end:06d}"
            chunk_ids.append(chunk_id)
            chunk_docs.append(text)
            chunk_metas.append(
                {
                    "start_idx": chunk_msgs[0].global_idx,
                    "end_idx": chunk_msgs[-1].global_idx,
                    "row_min": min(m.row_idx for m in chunk_msgs),
                    "row_max": max(m.row_idx for m in chunk_msgs),
                }
            )
            if end == len(messages):
                break

        # Batch in groups to avoid oversized single calls on large corpora
        batch_size = 256
        embeddings = self.embedder.encode(chunk_docs, show_progress_bar=False, batch_size=64).tolist()
        for i in range(0, len(chunk_ids), batch_size):
            self.chunk_collection.upsert(
                ids=chunk_ids[i:i + batch_size],
                documents=chunk_docs[i:i + batch_size],
                metadatas=chunk_metas[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
            )

    # ---------- RETRIEVAL ----------

    def retrieve(self, query: str, top_k_summaries: int = 3, top_k_chunks: int = 5) -> List[RetrievedItem]:
        q_emb = self.embedder.encode([query], show_progress_bar=False).tolist()

        results: List[RetrievedItem] = []

        summary_res = self.summary_collection.query(
            query_embeddings=q_emb, n_results=top_k_summaries
        )
        for doc, meta, dist in zip(
            summary_res["documents"][0], summary_res["metadatas"][0], summary_res["distances"][0]
        ):
            results.append(RetrievedItem("topic_summary", doc, meta, dist))

        chunk_res = self.chunk_collection.query(
            query_embeddings=q_emb, n_results=top_k_chunks
        )
        for doc, meta, dist in zip(
            chunk_res["documents"][0], chunk_res["metadatas"][0], chunk_res["distances"][0]
        ):
            results.append(RetrievedItem("message_chunk", doc, meta, dist))

        return results

    @staticmethod
    def build_context_block(items: List[RetrievedItem]) -> str:
        """Combine retrieved summaries + chunks into one context string
        for the chatbot's answer-generation prompt."""
        summaries = [i for i in items if i.source == "topic_summary"]
        chunks = [i for i in items if i.source == "message_chunk"]

        parts = []
        if summaries:
            parts.append("RELEVANT TOPIC SUMMARIES:")
            for s in summaries:
                parts.append(f"- {s.text}")
        if chunks:
            parts.append("\nRELEVANT RAW MESSAGE EXCERPTS:")
            for c in chunks:
                parts.append(f"- {c.text}")
        return "\n".join(parts)


if __name__ == "__main__":
    from data_loader import load_messages
    from topic_segmenter import TopicSegmenter
    from checkpoint_summarizer import summarize_all

    msgs = load_messages("data/conversations.csv")
    sample = msgs[:300]

    seg = TopicSegmenter()
    topic_segments = seg.segment_by_topic(sample)
    fixed_segments = seg.segment_fixed_100(sample)

    topic_summaries = summarize_all(topic_segments)
    fixed_summaries = summarize_all(fixed_segments)

    store = RagStore()
    store.ingest_summaries(topic_summaries + fixed_summaries)
    store.ingest_message_chunks(sample)

    results = store.retrieve("What city is someone moving to?")
    for r in results:
        print(f"[{r.source}] dist={r.distance:.3f} :: {r.text[:100]}")
