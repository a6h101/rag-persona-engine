# Conversation RAG + Persona Chatbot

Built for the KaStack Labs AI/ML Engineer Intern assessment.

## What this is

A local-first (no external LLM API) pipeline that:
1. Splits a long conversation stream into topic checkpoints and fixed
   100-message checkpoints, summarizing each.
2. Indexes both summaries and raw message chunks in ChromaDB for
   retrieval.
3. Extracts a structured, evidence-grounded persona (habits, facts,
   traits, communication style) as JSON.
4. Serves a Streamlit chatbot that answers questions about the user by
   combining RAG retrieval with the persona profile.

## A note on the dataset

`conversations.csv` (as provided) has no header and no timestamp
column. Each row is one independent, short two-person conversation
(`User 1` / `User 2` turns, 6-67 turns per row, ~11,000 rows total).
The role labels are local to each row, i.e. "User 1" in row 1 and
"User 1" in row 5000 are not the same person.

This doesn't match a literal "one continuous user, one message per
day" reading. To still satisfy the task's explicit requirement to
"process conversations in chronological order" and build checkpoints
across hundreds of messages, we concatenate all rows in their original
CSV order into one continuous message stream (preserving each
message's original row and turn index as metadata). This is a
deliberate, documented adaptation given the data's actual shape, not
an attempt to hide a mismatch. Persona extraction is done over a
chosen speaker slice (default: all "User 1" turns pooled across the
processed rows); the code also supports a single-row persona if a true
single-user profile is wanted instead.

## How topic-change detection works

`topic_segmenter.py` embeds every message with `all-MiniLM-L6-v2`
(sentence-transformers, fully local, no API calls). It keeps a
running centroid (mean embedding) of the current topic segment. For
each new message, it computes cosine similarity between that message
and the centroid. If similarity drops below a threshold (default
0.45) and the current segment already has at least `min_segment_len`
messages (default 5, to avoid 1-message noise), the segment is closed
as a topic checkpoint and a new one starts at that message. This is a
standard, lightweight embedding-drift approach to topic segmentation,
similar in spirit to TextTiling, adapted to use sentence embeddings
instead of lexical overlap.

Independently, `segment_fixed_100` chunks the same stream into fixed
blocks of 100 messages regardless of topic, as a separate checkpoint
scheme.

Each resulting segment (topic or fixed-100) is summarized by a local
Ollama model (`llama3.2`) via `checkpoint_summarizer.py`. If Ollama
isn't reachable, the code falls back to an extractive summary instead
of crashing.

## How retrieval works

`rag_store.py` maintains two ChromaDB collections:
- `topic_summaries`: one entry per topic/fixed-100 checkpoint, embedded
  from its summary text.
- `message_chunks`: overlapping windows of raw messages (10 messages
  per chunk, stride 5), embedded from raw text.

A query is embedded once and searched against both collections. The
top-k summary hits give broad orientation ("what general topic is
this about"); the top-k chunk hits give grounded, specific evidence
(exact original lines). Both are concatenated into a single context
block (summaries first, then raw excerpts) that's passed to the
chatbot's answer-generation prompt — this is the "combine both to
generate the answer" step the task asks for.

## How persona extraction works

`persona_extractor.py` scans every message from a target speaker
against keyword/regex patterns across four categories: habits,
personal facts, personality traits, and communication style. Every
hit stores the literal evidence line and message index, so every
claim in the final JSON can be traced back to an actual quote — nothing
is inferred without a backing line. Communication style (avg message
length, exclamation/question frequency, emoji use) is computed
directly from message statistics, not pattern matching.

An optional second pass asks a local Ollama model to turn a category's
evidence lines into a clean one-line label (e.g. several yoga/running
mentions -> "physically active"), but the model is only given the
already-extracted evidence and is instructed not to introduce new
facts. This keeps the persona grounded in actual conversation signals
per the task's requirement, while still producing readable output.

## Setup

```bash
pip install -r requirements.txt
# Ensure Ollama is running locally with llama3.2 pulled:
ollama pull llama3.2
ollama serve
```

Place `conversations.csv` in `data/`.

## Running

```bash
# 1. Build the index + persona (one-time, or whenever data changes)
python build_index.py

# 2. Launch the chatbot
streamlit run chatbot.py
```

`build_index.py` defaults to processing the first 2,000 messages for
fast demo turnaround (full-corpus Ollama summarization over ~191k
messages and ~1,900+ checkpoints is the same logic, just longer
runtime). Set `MAX_MESSAGES = None` in `build_index.py` to run the
full file.

## Project structure

```
data_loader.py            # CSV -> flat chronological Message stream
topic_segmenter.py         # topic-drift + fixed-100 checkpointing
checkpoint_summarizer.py   # local Ollama summarization per checkpoint
rag_store.py               # ChromaDB ingestion + combined retrieval
persona_extractor.py       # evidence-grounded persona JSON extraction
chatbot.py                 # Streamlit UI tying RAG + persona together
build_index.py             # one-shot pipeline runner
```

## Design tradeoffs / what I'd improve with more time

- Topic threshold (0.45) was set heuristically; with more time I'd
  validate it against a small hand-labeled set of true topic
  boundaries and tune precision/recall instead of eyeballing it.
- Persona patterns are keyword/regex based for speed and auditability.
  A learned classifier or LLM-based extraction (still constrained to
  cite evidence) would generalize better beyond the patterns I wrote.
- Chunking uses a fixed window/stride; sentence- or turn-aware
  chunking with overlap-aware deduplication would reduce redundant
  retrieval hits.
