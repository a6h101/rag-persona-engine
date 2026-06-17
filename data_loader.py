"""
data_loader.py

Loads conversations.csv and flattens it into a single chronological
message stream.

DATA SHAPE NOTE (important, also explained in README):
The provided CSV has no header row. Each row is one INDEPENDENT short
two-person conversation (avg ~17 turns, range 6-67), formatted as:

    User 1: ...
    User 2: ...
    User 1: ...
    ...

"User 1" / "User 2" are generic role labels per-conversation, not a
single tracked identity across rows (row 1's "User 1" and row 500's
"User 1" are different people). There is no timestamp column.

To satisfy the assessment's requirement of "process conversations in
chronological order" and build long-range topic/100-message checkpoints,
we concatenate all rows in CSV row order into one continuous message
stream, preserving original row index and turn index as metadata. This
is a deliberate, documented adaptation given the dataset's actual shape.

For persona extraction we treat each (row_index, speaker_label) pair as
a distinct "persona slice", and additionally support pooling all slices
that share a speaker_label if a per-corpus aggregate persona is wanted.
"""

import csv
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    global_idx: int      # 0-based position in the full concatenated stream
    row_idx: int          # which CSV row (conversation) this came from
    turn_idx: int          # 0-based position within that row's conversation
    speaker: str           # "User 1" or "User 2" (label local to that row)
    text: str


SPEAKER_LINE_RE = re.compile(r"^(User\s*\d+):\s*(.*)$")


def parse_row_into_turns(row_text: str) -> List[tuple]:
    """Split one CSV row's raw text into (speaker, text) turns."""
    turns = []
    for line in row_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_LINE_RE.match(line)
        if m:
            speaker, text = m.group(1), m.group(2).strip()
            turns.append((speaker, text))
        else:
            # Continuation of previous turn (wrapped line) — append to last
            if turns:
                speaker, prev_text = turns[-1]
                turns[-1] = (speaker, (prev_text + " " + line).strip())
    return turns


def load_messages(csv_path: str) -> List[Message]:
    """Load the full CSV and return a flat, globally-ordered message list."""
    messages: List[Message] = []
    global_idx = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            raw_text = row[0]
            turns = parse_row_into_turns(raw_text)
            for turn_idx, (speaker, text) in enumerate(turns):
                if not text:
                    continue
                messages.append(
                    Message(
                        global_idx=global_idx,
                        row_idx=row_idx,
                        turn_idx=turn_idx,
                        speaker=speaker,
                        text=text,
                    )
                )
                global_idx += 1

    return messages


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/conversations.csv"
    msgs = load_messages(path)
    print(f"Loaded {len(msgs)} messages from {path}")
    print("First 5 messages:")
    for m in msgs[:5]:
        print(f"  [{m.global_idx}] row={m.row_idx} turn={m.turn_idx} {m.speaker}: {m.text}")
    print("...")
    print(f"Total conversations (rows): {msgs[-1].row_idx + 1}")
