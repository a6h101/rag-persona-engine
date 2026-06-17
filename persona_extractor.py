"""
persona_extractor.py

Extracts a structured persona JSON for a given speaker from their
actual messages. Every field is grounded in literal evidence lines
(stored alongside each extracted item), per the task's explicit rule:
"Persona should be based on actual conversation signals, not guesses."

Four categories:
  - habits: routines/lifestyle signals (sleep, food, exercise, etc.)
  - personal_facts: relationships, events, jobs, locations mentioned
  - personality_traits: signals like humor, emotional expressiveness
  - communication_style: message length, tone markers, emoji/punctuation use

APPROACH:
  Stage 1 (deterministic, regex/keyword-based): scans every message
  from the target speaker for category-relevant keyword patterns and
  records the literal evidence sentence. This is fast, fully local,
  and auditable, every claim can be traced to a quoted line.

  Stage 2 (Ollama-assisted, optional): the evidence lines collected in
  stage 1 are handed to a local LLM ONLY to phrase a clean one-line
  label per signal (e.g. turning 3 evidence lines about cars into the
  label "car enthusiast"). The LLM is never allowed to introduce facts
  that aren't in the evidence it was given.

Because this dataset is ~11,000 independent short conversations rather
than one continuous user, persona extraction here operates on a
SPEAKER SLICE: either a single (row_idx, speaker_label) for a per-
conversation persona, or a pooled set of slices for an aggregate
persona across many conversations. See README for details.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict

from data_loader import Message

try:
    from checkpoint_summarizer import _call_ollama
except ImportError:
    _call_ollama = None


HABIT_PATTERNS = {
    "sleep_schedule": [r"\bwake up\b", r"\bstay up\b", r"\bnight owl\b", r"\bearly bird\b", r"\bsleep\b"],
    "diet_food": [r"\bvegan\b", r"\bvegetarian\b", r"\blove(?:s)? (?:to )?eat\b", r"\bfavorite food\b", r"\bcook(?:ing)?\b", r"\bbaking\b"],
    "exercise": [r"\byoga\b", r"\brun(?:ning)?\b", r"\bgym\b", r"\bworkout\b", r"\bexercise\b"],
    "substance_routine": [r"\bcoffee\b", r"\bsmoking\b", r"\bdrink(?:ing)?\b"],
}

FACT_PATTERNS = {
    "relationships": [r"\bmy (?:wife|husband|girlfriend|boyfriend|partner|fianc[ée]+)\b",
                       r"\bmarried\b", r"\bdating\b", r"\bmy (?:mom|dad|parents|sister|brother|kids?|children)\b"],
    "occupation_education": [r"\bI (?:work|study|major in)\b", r"\bstudent\b", r"\bcollege\b",
                              r"\bschool\b", r"\bjob\b", r"\bcareer\b"],
    "life_events": [r"\bmoving to\b", r"\bjust (?:got|started|finished)\b", r"\bgraduat(?:ed|ing)\b",
                     r"\bnew (?:job|house|city|apartment)\b"],
    "location": [r"\bI (?:live|grew up|am from|moved)\b"],
}

TRAIT_PATTERNS = {
    "humor": [r"\bhaha\b", r"\blol\b", r"\bjoke\b", r"\bfunny\b"],
    "emotional_expressiveness": [r"\bI feel\b", r"\bI'm (?:so|really) (?:happy|sad|excited|nervous|worried)\b",
                                   r"\blove\b", r"\bafraid\b", r"\bexcited\b"],
    "seriousness": [r"\bhonestly\b", r"\bto be fair\b", r"\bin my opinion\b"],
    "enthusiasm": [r"!{1,}"],
}


@dataclass
class EvidenceItem:
    category: str
    signal: str
    evidence_text: str
    message_global_idx: int


@dataclass
class PersonaProfile:
    speaker_label: str
    source_rows: List[int]
    habits: Dict[str, List[str]]
    personal_facts: Dict[str, List[str]]
    personality_traits: Dict[str, List[str]]
    communication_style: Dict[str, object]
    evidence_log: List[dict]


def _scan_patterns(messages: List[Message], pattern_dict: Dict[str, List[str]]) -> Dict[str, List[EvidenceItem]]:
    hits: Dict[str, List[EvidenceItem]] = defaultdict(list)
    for msg in messages:
        text_lower = msg.text.lower()
        for category, patterns in pattern_dict.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    hits[category].append(
                        EvidenceItem(
                            category=category,
                            signal=pat,
                            evidence_text=msg.text,
                            message_global_idx=msg.global_idx,
                        )
                    )
                    break  # avoid double-counting same message for same category
    return hits


def _analyze_communication_style(messages: List[Message]) -> Dict[str, object]:
    if not messages:
        return {}

    lengths = [len(m.text.split()) for m in messages]
    avg_len = sum(lengths) / len(lengths)

    exclamations = sum(m.text.count("!") for m in messages)
    questions = sum(m.text.count("?") for m in messages)
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
    )
    emoji_count = sum(len(emoji_pattern.findall(m.text)) for m in messages)

    if avg_len <= 6:
        length_label = "short, concise messages"
    elif avg_len <= 15:
        length_label = "medium-length messages"
    else:
        length_label = "long, detailed messages"

    tone_markers = []
    if exclamations / max(len(messages), 1) > 0.3:
        tone_markers.append("frequent exclamation marks (enthusiastic tone)")
    if questions / max(len(messages), 1) > 0.3:
        tone_markers.append("asks frequent questions (engaged/curious)")
    if emoji_count > 0:
        tone_markers.append(f"uses emoji ({emoji_count} found)")

    return {
        "avg_words_per_message": round(avg_len, 1),
        "length_label": length_label,
        "exclamation_count": exclamations,
        "question_count": questions,
        "emoji_count": emoji_count,
        "tone_markers": tone_markers,
    }


def extract_persona(messages: List[Message], speaker_label: str, use_llm_labels: bool = False) -> PersonaProfile:
    """
    messages: ALL messages already filtered to a single speaker
              (e.g. only "User 1" turns from one or more rows).
    speaker_label: "User 1" or "User 2" (the role label being profiled).
    """
    habit_hits = _scan_patterns(messages, HABIT_PATTERNS)
    fact_hits = _scan_patterns(messages, FACT_PATTERNS)
    trait_hits = _scan_patterns(messages, TRAIT_PATTERNS)
    comm_style = _analyze_communication_style(messages)

    evidence_log: List[dict] = []

    def collapse(hits: Dict[str, List[EvidenceItem]]) -> Dict[str, List[str]]:
        out = {}
        for category, items in hits.items():
            out[category] = [it.evidence_text for it in items[:5]]  # cap evidence shown
            for it in items:
                evidence_log.append(asdict(it))
        return out

    habits = collapse(habit_hits)
    facts = collapse(fact_hits)
    traits = collapse(trait_hits)

    if use_llm_labels and _call_ollama is not None:
        habits = _llm_label_pass(habits, "habit")
        traits = _llm_label_pass(traits, "personality trait")

    source_rows = sorted(set(m.row_idx for m in messages))

    return PersonaProfile(
        speaker_label=speaker_label,
        source_rows=source_rows,
        habits=habits,
        personal_facts=facts,
        personality_traits=traits,
        communication_style=comm_style,
        evidence_log=evidence_log,
    )


def _llm_label_pass(category_evidence: Dict[str, List[str]], kind: str) -> Dict[str, List[str]]:
    """Optional: ask local LLM to produce a one-line human-readable label
    per category, STRICTLY from the evidence lines given (no new facts)."""
    labeled = {}
    for category, evidence_lines in category_evidence.items():
        if not evidence_lines:
            continue
        evidence_block = "\n".join(f"- {e}" for e in evidence_lines)
        prompt = (
            f"Based ONLY on these evidence lines, write a single short {kind} "
            f"label (3-6 words) that describes this person. Do not invent "
            f"anything not implied by the lines.\n\n{evidence_block}\n\nLabel:"
        )
        try:
            label = _call_ollama(prompt)
        except Exception:
            label = category.replace("_", " ")
        labeled[category] = {"label": label, "evidence": evidence_lines}
    return labeled


def save_persona(persona: PersonaProfile, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(persona), f, indent=2)


if __name__ == "__main__":
    from data_loader import load_messages

    msgs = load_messages("data/conversations.csv")
    # Example: build a pooled persona for "User 2" across first 50 rows
    target_rows = set(range(0, 50))
    speaker_msgs = [m for m in msgs if m.row_idx in target_rows and m.speaker == "User 2"]

    persona = extract_persona(speaker_msgs, speaker_label="User 2 (rows 0-49 pooled)")
    save_persona(persona, "outputs/sample_persona.json")
    print(json.dumps(asdict(persona), indent=2)[:2000])
