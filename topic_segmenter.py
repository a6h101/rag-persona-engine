"""
topic_segmenter.py

Two independent checkpointing schemes over the chronological message stream:

1. TOPIC checkpoints (semantic, variable-length)
   We embed each message with a sentence-transformer. We maintain a
   rolling "current topic centroid" = mean embedding of messages seen
   since the last checkpoint. For each new message, we compute cosine
   similarity to that centroid. If similarity drops below a threshold
   (i.e. the new message is semantically far from what's been discussed),
   AND a minimum segment length has been satisfied (to avoid noisy
   1-message "topics"), we close the current segment as a topic
   checkpoint and start a new one.

   This is a standard, lightweight, fully local approach to topic
   segmentation (no LLM call needed for the *detection* step — only
   for summarizing the resulting segment afterward). It's the same
   family of technique used in TextTiling-style and embedding-based
   topic segmentation methods.

2. FIXED checkpoints (every 100 messages, regardless of topic)
   Simple positional chunking: messages [0:100), [100:200), etc.
   Independent of topic boundaries, as the task requires.

Both produce a list of (start_idx, end_idx, message_list) segments that
get summarized by checkpoint_summarizer.py.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import Message


@dataclass
class Segment:
    start_idx: int
    end_idx: int  # exclusive
    messages: List[Message]
    kind: str  # "topic" or "fixed100"


class TopicSegmenter:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.45,
        min_segment_len: int = 5,
    ):
        """
        similarity_threshold: below this cosine similarity to the running
            topic centroid, a message is considered a topic shift.
            Tune lower (e.g. 0.3) for fewer, broader topics; higher
            (e.g. 0.55) for more granular topic splits.
        min_segment_len: minimum messages before we allow a new topic
            checkpoint to fire, prevents 1-2 message noisy segments.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_segment_len = min_segment_len

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, batch_size=64)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def segment_by_topic(self, messages: List[Message]) -> List[Segment]:
        if not messages:
            return []

        embeddings = self._embed([m.text for m in messages])

        segments: List[Segment] = []
        seg_start = 0
        centroid = embeddings[0].copy()
        centroid_count = 1

        for i in range(1, len(messages)):
            sim = self._cosine_sim(embeddings[i], centroid)
            current_len = i - seg_start

            if sim < self.similarity_threshold and current_len >= self.min_segment_len:
                # Close current segment as a topic checkpoint
                segments.append(
                    Segment(
                        start_idx=seg_start,
                        end_idx=i,
                        messages=messages[seg_start:i],
                        kind="topic",
                    )
                )
                # Start new segment/topic at this message
                seg_start = i
                centroid = embeddings[i].copy()
                centroid_count = 1
            else:
                # Still same topic — update rolling centroid (running mean)
                centroid = (centroid * centroid_count + embeddings[i]) / (centroid_count + 1)
                centroid_count += 1

        # Final trailing segment
        if seg_start < len(messages):
            segments.append(
                Segment(
                    start_idx=seg_start,
                    end_idx=len(messages),
                    messages=messages[seg_start:],
                    kind="topic",
                )
            )

        return segments

    @staticmethod
    def segment_fixed_100(messages: List[Message]) -> List[Segment]:
        segments = []
        for start in range(0, len(messages), 100):
            end = min(start + 100, len(messages))
            segments.append(
                Segment(
                    start_idx=start,
                    end_idx=end,
                    messages=messages[start:end],
                    kind="fixed100",
                )
            )
        return segments


if __name__ == "__main__":
    from data_loader import load_messages

    msgs = load_messages("data/conversations.csv")
    # Use a manageable slice for a quick smoke test
    sample = msgs[:500]

    seg = TopicSegmenter()
    topic_segments = seg.segment_by_topic(sample)
    fixed_segments = seg.segment_fixed_100(sample)

    print(f"Sample size: {len(sample)} messages")
    print(f"Topic segments found: {len(topic_segments)}")
    for s in topic_segments:
        print(f"  Topic [{s.start_idx}:{s.end_idx}] ({s.end_idx - s.start_idx} msgs) "
              f"first_msg='{s.messages[0].text[:50]}'")
    print(f"Fixed-100 segments found: {len(fixed_segments)}")
