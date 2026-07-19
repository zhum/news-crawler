#!/usr/bin/env python3
"""
digest_email_builder.py — mechanical half of the weekly-hpc-episode-ideas
routine. Import this in a single Cowork code-execution step.

Division of labor:
  - This script: parsing digest/sheet/outline text, flagging likely
    duplicates via fuzzy title matching, rendering the final email.
  - Claude (writing the enrichment inline, informed by actually reading
    the digest): deciding which flagged items are real duplicates vs.
    fresh angles, and writing the angle / why-now text for survivors.

Typical usage inside one Cowork code-execution call:

    from digest_email_builder import parse_digest_md, parse_sheet_csv, \\
        parse_outline_topics, flag_overlaps, render_email

    digest_items = parse_digest_md(digest_text)       # from Drive read
    past = parse_sheet_csv(sheet_csv_text)            # from Drive read
    outline_topics = parse_outline_topics(outline_text)  # from Drive read

    candidates = flag_overlaps(digest_items, past, outline_topics)

    # Claude writes this part inline, based on actually reading candidates
    # and the digest — this is the judgment step, not scriptable:
    enriched = [
        {
            **c,
            "angle": "...",       # Claude writes this
            "why_now": "...",     # Claude writes this
            "decision": "new",    # "new" | "already_planned" | "drop"
        }
        for c in candidates
        # Claude decides which to keep/drop/mark already_planned here
    ]

    subject, plain_body, html_body = render_email(enriched)
    # -> pass to Gmail:create_draft
"""

import csv
import io
import re
from datetime import date
from difflib import SequenceMatcher


# --- Parsing ---------------------------------------------------------

def parse_digest_md(text):
    """Parse export_digest.py's markdown output into a list of dicts:
    {title, rating, tags, source, summary}. Skips the arXiv section
    (no rating) by only picking up items under headers other than
    'ArXiv papers' -- arXiv items still get {title, source, summary}
    with rating=None.
    """
    items = []
    blocks = re.split(r"\n(?=### )", text)
    current_section = None
    for block in text.split("\n"):
        if block.startswith("## "):
            current_section = block[3:].strip()

    for block in blocks:
        block = block.strip()
        if not block.startswith("### "):
            continue
        title_match = re.match(r"### (.+)", block)
        title = title_match.group(1).strip() if title_match else "Untitled"
        rating_match = re.search(r"Rating:\s*(\d+)/10", block)
        tags_match = re.search(r"Tags:\s*(.+)", block)
        source_match = re.search(r"Source:\s*(\S+)", block)
        # summary: everything after the last "- " metadata line
        lines = block.split("\n")
        summary_lines = []
        past_meta = False
        for line in lines[1:]:
            if line.strip().startswith("- "):
                continue
            if line.strip():
                summary_lines.append(line.strip())
        summary = " ".join(summary_lines).strip()

        items.append({
            "title": title,
            "rating": int(rating_match.group(1)) if rating_match else None,
            "tags": tags_match.group(1).strip() if tags_match else "",
            "source": source_match.group(1).strip() if source_match else "",
            "summary": summary,
        })
    return items


def parse_sheet_csv(csv_text):
    """Parse the episode index sheet (exported/read as CSV text) into
    a list of {title, status} dicts. Expects columns including
    'Title' and 'Status' (case-insensitive, order-independent)."""
    reader = csv.DictReader(io.StringIO(csv_text))
    out = []
    for row in reader:
        # normalize header case
        norm = {k.strip().lower(): v for k, v in row.items() if k}
        title = norm.get("title", "").strip()
        status = norm.get("status", "").strip().lower()
        if title:
            out.append({"title": title, "status": status or "planned"})
    return out


def parse_outline_topics(text):
    """Best-effort extraction of topic lines from the free-form outline
    doc: markdown headers and bullet points, treated as 'planned'-
    equivalent topics for dedup purposes."""
    topics = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^(#{1,6}\s+|[-*]\s+)(.+)", line)
        if m:
            topic = m.group(2).strip()
            if len(topic) > 8:  # skip trivially short lines
                topics.append(topic)
    return topics


# --- Fuzzy dedup flagging --------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "for", "and", "or", "of", "in", "on", "to", "with",
    "your", "how", "what", "why", "is", "are", "vs", "at", "into",
}


def _tokens(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOPWORDS}


def _word_overlap(a, b):
    """Jaccard similarity on significant words — catches semantic overlap
    ('Slurm', 'QoS' in both) that pure character matching misses."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _combined_score(a, b):
    """Blend of character-sequence similarity and word overlap. Neither
    signal alone is reliable: character matching can score unrelated
    titles highly on shared substrings/word lengths, while word overlap
    alone misses paraphrases. Taking the max of both, weighted toward
    word overlap, catches more real duplicates without over-trusting
    either signal."""
    char_score = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    word_score = _word_overlap(a, b)
    return max(word_score * 1.1, char_score * 0.7, char_score)


def flag_overlaps(digest_items, past_episodes, outline_topics, threshold=0.3, top_n=3):
    """For each digest item, find the top-N candidate matches (by
    combined score) against past episodes + outline topics, and attach
    them if any clear the threshold. This is a rough signal for Claude
    to review, not a verdict — always look at all top_n candidates
    before deciding drop vs. keep vs. already_planned; the single
    highest-scoring match is not guaranteed to be the semantically
    correct one. Unflagged items (empty possible_overlaps) can usually
    be trusted as genuinely new without further checking.
    """
    out = []
    all_past = [(ep["title"], ep["status"]) for ep in past_episodes] + \
               [(t, "planned") for t in outline_topics]

    for item in digest_items:
        scored = [
            (title, status, _combined_score(item["title"], title))
            for title, status in all_past
        ]
        scored = [s for s in scored if s[2] >= threshold]
        scored.sort(key=lambda s: -s[2])

        enriched_item = dict(item)
        enriched_item["possible_overlaps"] = [
            {"matched_title": t, "matched_status": st, "score": round(sc, 2)}
            for t, st, sc in scored[:top_n]
        ]
        out.append(enriched_item)
    return out


# --- Rendering ---------------------------------------------------------

def render_email(enriched_items, week_of=None):
    """enriched_items: list of dicts from flag_overlaps, each with
    'angle', 'why_now', and 'decision' added by Claude ('new',
    'already_planned', or 'drop'). Items with decision == 'drop' are
    excluded from the output but still counted in the receipt.
    Returns (subject, plain_body, html_body).
    """
    week_of = week_of or date.today().isoformat()
    scanned = len(enriched_items)
    dropped = [i for i in enriched_items if i.get("decision") == "drop"]
    kept = [i for i in enriched_items if i.get("decision") != "drop"]

    subject = f"HPC Channel — Episode Ideas — Week of {week_of}"

    receipt = (
        f"Scanned: {scanned} items. "
        f"Filtered as duplicates: {len(dropped)}. "
        f"Final shortlist: {len(kept)}."
    )

    plain_lines = [receipt, ""]
    html_parts = [f"<p>{receipt}</p>"]

    for idx, item in enumerate(kept, start=1):
        overlap_note = "none"
        overlaps = item.get("possible_overlaps") or []
        if item.get("decision") == "already_planned" and overlaps:
            m = overlaps[0]
            overlap_note = f"already planned (\"{m.get('matched_title', '?')}\")"
        elif overlaps:
            m = overlaps[0]
            overlap_note = f"related to \"{m.get('matched_title', '?')}\" ({m.get('matched_status', '?')}), but fresh angle"

        plain_lines += [
            f"{idx}. {item['title']}",
            f"Source: {item.get('source', '')}",
            f"Angle: {item.get('angle', '')}",
            f"Why now: {item.get('why_now', '')}",
            f"Overlap note: {overlap_note}",
            "",
        ]
        html_parts.append(
            f"<h3>{idx}. {item['title']}</h3>"
            f"<p>Source: <a href=\"{item.get('source', '')}\">{item.get('source', '')}</a></p>"
            f"<p>Angle: {item.get('angle', '')}</p>"
            f"<p>Why now: {item.get('why_now', '')}</p>"
            f"<p>Overlap note: {overlap_note}</p>"
        )

    plain_body = "\n".join(plain_lines)
    html_body = "\n".join(html_parts)
    return subject, plain_body, html_body
