#!/usr/bin/env python3
"""
generate_digest_email.py — fully automated version of the weekly episode-
ideas pipeline, meant to run from cron with no human/Claude-session
approval steps.

Division of labor (same as the Cowork version, just with the judgment
step now going through the API instead of a chat session):
  - digest_email_builder.py: parsing, fuzzy dedup flagging, HTML render
  - Anthropic API (this script): the actual judgment call — reading each
    candidate + its possible_overlaps and deciding drop/keep/angle
  - Gmail API (this script): creating the draft, and optionally sending it
    (only if --send is passed — draft-only is still the default)

Prerequisites:
    pip install anthropic google-auth google-auth-oauthlib google-api-python-client

    export ANTHROPIC_API_KEY=sk-ant-...

    One-time Gmail OAuth setup (see gmail_oauth_setup.py in this repo):
    creates gmail_token.json, which this script then reuses headlessly.

Usage:
    ./generate_digest_email.py \
        --digest-dir digests \
        --sheet-csv drive-cache/episode-sheet.csv \
        --outline-txt drive-cache/outline.txt \
        --to you@example.com
"""

import argparse
import base64
import getpass
import json
import os
import re
import signal
import smtplib
import subprocess
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from digest_email_builder import (  # noqa: E402
    parse_digest_md, parse_sheet_csv, parse_outline_topics,
    flag_overlaps, render_email,
)

ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def latest_digest_file(digest_dir):
    files = sorted(glob(os.path.join(digest_dir, "*-hpc-digest.md")))
    return files[-1] if files else None


def last_run_marker_path(digest_dir):
    return os.path.join(digest_dir, ".last-emailed-digest")


def already_processed(digest_path, digest_dir):
    marker = last_run_marker_path(digest_dir)
    if not os.path.exists(marker):
        return False
    with open(marker, "r", encoding="utf-8") as f:
        return f.read().strip() == os.path.basename(digest_path)


def mark_processed(digest_path, digest_dir):
    with open(last_run_marker_path(digest_dir), "w", encoding="utf-8") as f:
        f.write(os.path.basename(digest_path))


# --- Judgment step via Anthropic API ------------------------------------

def build_judgment_prompt(candidates):
    """candidates: output of flag_overlaps(). Ask Claude for a JSON array,
    same length/order as input, one decision object per candidate."""
    candidate_json = json.dumps([
        {
            "index": i,
            "title": c["title"],
            "summary": c.get("summary", ""),
            "tags": c.get("tags", ""),
            "possible_overlaps": c.get("possible_overlaps", []),
        }
        for i, c in enumerate(candidates)
    ], indent=2)

    return f"""You are helping decide which HPC news items make good YouTube \
episode ideas for a sysadmin-focused channel (topics: scheduling, storage, \
networking, benchmarking, systems administration).

For each candidate below, decide:
- "decision": "new" (genuinely new topic), "already_planned" (matches a \
planned/outline topic in possible_overlaps AND the angle isn't meaningfully \
different), or "drop" (genuine duplicate of an AIRED episode in \
possible_overlaps, or too thin to support a real episode)
- "angle": for "new" or "already_planned" with a fresh angle, a specific \
take (not just "cover this news") — a couple sentences. Empty string for "drop".
- "why_now": the timeliness hook, a short sentence. Empty string for "drop".
- "priority": for "new" or "already_planned" items only, an integer 1-10 \
rating how compelling this is as THIS WEEK'S standout episode idea, \
considering timeliness and likely audience interest. 0 for "drop".

IMPORTANT: possible_overlaps is a rough fuzzy-matching signal, not a \
verdict — the highest-scored match is not guaranteed to be the semantically \
correct one. Look at all of them, and use your own judgment about whether \
the candidate is really a duplicate.

Candidates:
{candidate_json}

Respond with ONLY a JSON array, no other text, no markdown code fences. \
Same length and order as the input, one object per candidate:
[{{"index": 0, "decision": "new", "angle": "...", "why_now": "...", "priority": 7}}, ...]
"""


def call_anthropic(prompt, api_key):
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {e.code}: {detail}") from e


def generate_idea_anthropic(past_titles, news_titles, api_key):
    """Generate a weekly episode idea via the Anthropic API."""
    prompt = (
        "We're brainstorming ideas for a YouTube channel about HPC system "
        "administration and HPC basics. Generate a fresh, high-impact idea "
        "for a new episode, inspired by this week's news: "
        f"{news_titles}. No preamble. Do not repeat already aired topics: "
        f"{past_titles}. If you struggle to generate an idea, tell which "
        "information you need to make it."
    )
    response = call_anthropic(prompt, api_key)
    text = extract_text(response)
    return text.strip() if text.strip() else ""


def markdown_to_html(text):
    """Convert basic markdown markup to HTML: headers, bold, italic,
    ordered/unordered lists, and paragraphs."""
    lines = text.split('\n')
    html_parts = []
    in_list = None  # 'ul' or 'ol', or None

    for line in lines:
        stripped = line.strip()

        # Empty line closes any open list and skips
        if not stripped:
            if in_list:
                html_parts.append(f'</{in_list}>')
                in_list = None
            continue

        # Headers: # H1 .. ###### H6
        header_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if header_match:
            if in_list:
                html_parts.append(f'</{in_list}>')
                in_list = None
            level = len(header_match.group(1))
            content = _md_inline(header_match.group(2))
            html_parts.append(f'<h{level}>{content}</h{level}>')
            continue

        # Unordered list item: - or * or +
        ul_match = re.match(r'^[-*+]\s+(.*)', stripped)
        if ul_match:
            if in_list != 'ul':
                if in_list:
                    html_parts.append(f'</{in_list}>')
                html_parts.append('<ul>')
                in_list = 'ul'
            content = _md_inline(ul_match.group(1))
            html_parts.append(f'<li>{content}</li>')
            continue

        # Ordered list item: 1. 2. etc.
        ol_match = re.match(r'^\d+\.\s+(.*)', stripped)
        if ol_match:
            if in_list != 'ol':
                if in_list:
                    html_parts.append(f'</{in_list}>')
                html_parts.append('<ol>')
                in_list = 'ol'
            content = _md_inline(ol_match.group(1))
            html_parts.append(f'<li>{content}</li>')
            continue

        # Plain text — paragraph
        if in_list:
            html_parts.append(f'</{in_list}>')
            in_list = None
        content = _md_inline(stripped)
        html_parts.append(f'<p>{content}</p>')

    if in_list:
        html_parts.append(f'</{in_list}>')

    return '\n'.join(html_parts)


def _md_inline(text):
    """Convert inline markdown: bold (**text** or __text__), italic
    (*text* or _text_), and bold+italic (***text*** or ___text___)."""
    # Bold+italic: ***text*** or ___text___
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'___(.+?)___', r'<strong><em>\1</em></strong>', text)
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', text)
    return text


def generate_idea_local(ai_summary, model, past_titles, news_titles):
    """Generate a weekly episode idea via a local GGUF model."""
    idea_prompt = ai_summary.render_chat_prompt(
        _JUDGE_SYSTEM_PROMPT,
        f"We're brainstorming ideas for the podcast about HPC system "
        f"administration and HPC basics. Generate a fresh, high-impact "
        f"idea for a new episode, inspired by this week's news: "
        f"{news_titles}. No preamble. Do not repeat already aired "
        f"topics, here is the list of aired topics: {past_titles}. "
        f"If you struggle to generate an idea, tell which information "
        f"you need to make it.",
        model.family)
    idea = model.generate(idea_prompt, max_new_tokens=2000).strip()
    # Strip reasoning blocks (e.g. Qwen <think>...</think>)
    idea = re.sub(r'<think>.*?</think>', '', idea, flags=re.DOTALL).strip()
    idea = markdown_to_html(idea)
    return idea if idea else ""


def extract_text(api_response):
    parts = [b["text"] for b in api_response.get("content", []) if b.get("type") == "text"]
    return "\n".join(parts)


def parse_judgment_json(text):
    """Defensive parse: strip markdown code fences if the model added
    them despite instructions not to."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def judge_candidates(candidates, api_key):
    if not candidates:
        return []
    prompt = build_judgment_prompt(candidates)
    response = call_anthropic(prompt, api_key)
    text = extract_text(response)
    try:
        judgments = parse_judgment_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"Could not parse judgment response as JSON: {e}\nRaw response:\n{text[:2000]}"
        ) from e

    by_index = {j["index"]: j for j in judgments if "index" in j}
    enriched = []
    for i, c in enumerate(candidates):
        j = by_index.get(i, {"decision": "drop", "angle": "", "why_now": "", "priority": 0})
        enriched.append({**c, **j})
    return enriched


# --- Judgment step via local GGUF model (llama-cpp-python) --------------

_JUDGE_SYSTEM_PROMPT = (
    "You help decide which HPC news items make good YouTube episode ideas "
    "for a sysadmin-focused channel (topics: scheduling, storage, "
    "networking, benchmarking, systems administration)."
)

_DECISION_GRAMMAR = 'root ::= "new" | "already_planned" | "drop"'


def _load_ai_summary_module():
    """Load ai-summary.py as a module (not a plain `import` — the filename
    has a hyphen). Requires the CLI block in that file to be guarded by
    `if __name__ == "__main__":`, otherwise this would fire its CLI logic
    against our sys.argv instead of loading cleanly."""
    import importlib.util
    path = Path(__file__).parent / "ai-summary.py"
    spec = importlib.util.spec_from_file_location("ai_summary_lib", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _overlaps_description(candidate):
    overlaps = candidate.get("possible_overlaps", [])
    if not overlaps:
        return "none"
    return "; ".join(
        f"{o['matched_title']} ({o['matched_status']}, score {o['score']})"
        for o in overlaps
    )


def load_local_model(model_path):
    """Load the GGUF model once, for reuse across both the judgment step
    and the scoring step below — reloading multi-GB weights twice per
    run would be wasteful."""
    ai_summary = _load_ai_summary_module()
    model = ai_summary.ModelWrapper(model_path)
    return ai_summary, model


def judge_candidates_local(candidates, ai_summary, model):
    """Same job as judge_candidates(), but using a local GGUF model via
    llama-cpp-python instead of the Anthropic API. Trades API cost/key
    management for weaker instruction-following, so unlike the batched
    single-call API version, this asks one small, narrowly-scoped
    question at a time (decision, then angle, then why-now) — matching
    the pattern ai-summary.py already uses for its own per-tag ratings.
    The decision is grammar-constrained to exactly one of three literal
    strings, so it can't come back malformed the way free-form JSON
    might with a smaller/quantized model."""
    if not candidates:
        return []

    enriched = []
    for c in candidates:
        overlaps_desc = _overlaps_description(c)
        base_facts = f"Title: {c['title']}\nSummary: {c.get('summary', '')}\n"

        decision_user = (
            base_facts +
            f"Possible overlaps with past/planned episodes (a rough fuzzy-"
            f"match signal, not a verdict — use your own judgment about "
            f"whether these are really related): {overlaps_desc}\n\n"
            "Decide: \"new\" if this is genuinely new, \"already_planned\" "
            "if it matches a planned/outline topic above AND the angle "
            "isn't meaningfully different, \"drop\" if it's a genuine "
            "duplicate of an AIRED episode above, or too thin to support "
            "a real episode.\nAnswer with exactly one word."
        )
        prompt = ai_summary.render_chat_prompt(
            _JUDGE_SYSTEM_PROMPT, decision_user, model.family)
        raw = model.generate(prompt, max_new_tokens=16, grammar_str=_DECISION_GRAMMAR)
        # Server mode ignores the grammar, and thinking models (Qwen) prefix
        # the answer with a <think>...</think> block — strip it first, or
        # every item falls through to the "drop" fail-safe below.
        decision = ai_summary._clean(raw).strip('"').lower()
        if decision not in ("new", "already_planned", "drop"):
            decision = "drop"  # fail safe: if the model didn't follow the
                                # grammar for some reason, don't email an
                                # item we're not confident about

        angle, why_now = "", ""
        if decision != "drop":
            angle_prompt = ai_summary.render_chat_prompt(
                _JUDGE_SYSTEM_PROMPT,
                base_facts + "\nWrite a specific episode angle (not just "
                "'cover this news'), 1-2 sentences, no preamble.",
                model.family)
            angle = ai_summary._clean(model.generate(angle_prompt, max_new_tokens=120))

            why_prompt = ai_summary.render_chat_prompt(
                _JUDGE_SYSTEM_PROMPT,
                base_facts + "\nIn one short sentence, why is this timely "
                "right now? No preamble.",
                model.family)
            why_now = ai_summary._clean(model.generate(why_prompt, max_new_tokens=60))

        enriched.append({**c, "decision": decision, "angle": angle, "why_now": why_now})

    return enriched


# --- Ranking (top-5 selection) -------------------------------------------
#
# Step 1: score each KEPT item individually, 1-10, via the same grammar-
#   constrained-integer pattern ai-summary.py already uses for tag ratings.
#   One item per call keeps each prompt tiny, so this stays well inside a
#   32K context regardless of how many items are kept — no context-window
#   pressure, unlike a single big "rank all N items" prompt would create.
# Step 2: sort by score in plain Python. No LLM call needed for this part —
#   sorting a list of numbers doesn't need a language model.

def score_candidates_local(kept_items, ai_summary, model):
    """Attach a 'priority' score (1-10) to each kept item in place, via
    one grammar-constrained call per item."""
    for item in kept_items:
        prompt = ai_summary.render_chat_prompt(
            _JUDGE_SYSTEM_PROMPT,
            f"Title: {item['title']}\n"
            f"Angle: {item.get('angle', '')}\n"
            f"Why now: {item.get('why_now', '')}\n\n"
            "Rate 1-10 how compelling this is as THIS WEEK'S standout "
            "episode idea for this channel, considering timeliness and "
            "likely audience interest. Answer with a single integer.",
            model.family)
        raw = model.generate(prompt, max_new_tokens=12, grammar_str=ai_summary.INT_GRAMMAR)
        try:
            score = int(re.search(r"\d+", ai_summary._clean(raw)).group())
            score = max(1, min(10, score))
        except (AttributeError, ValueError):
            score = 1  # fail safe: uncertain -> lowest priority, not highest
        item["priority"] = score
    return kept_items


# --- Gmail draft creation -------------------------------------------------

def create_gmail_draft(to_addr, subject, plain_body, html_body, token_path):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(
        token_path, scopes=["https://www.googleapis.com/auth/gmail.compose"]
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)

    message = MIMEMultipart("alternative")
    message["to"] = to_addr
    message["subject"] = subject
    message.attach(MIMEText(plain_body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}
    ).execute()
    return draft


def send_smtp_email(to_addr, from_addr, subject, plain_body, html_body,
                     host, port, user, password):
    message = MIMEMultipart("alternative")
    message["To"] = to_addr
    message["From"] = from_addr
    message["Subject"] = subject
    message.attach(MIMEText(plain_body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(host, port, timeout=60) as s:
        s.starttls()
        s.login(user, password)
        s.sendmail(from_addr, [to_addr], message.as_string())


def send_local_mail(to_addr, subject, html_body):
    """Fallback: hand the message to the local MTA via GNU mail / mailx.
    Returns the binary used, or None if neither is installed."""
    import shutil
    from email.header import Header
    for cmd in ("mail", "mailx"):
        if shutil.which(cmd):
            # -a appends a header in both GNU mailutils and bsd-mailx;
            # mail passes the subject raw, so RFC 2047-encode it (em-dashes)
            subprocess.run(
                [cmd, "-s", Header(subject, "utf-8").encode(),
                 "-a", "MIME-Version: 1.0",
                 "-a", "Content-Type: text/html; charset=UTF-8",
                 "-a", "Content-Transfer-Encoding: 8bit", to_addr],
                input=html_body, text=True, check=True, timeout=120)
            return cmd
    return None


def notify_desktop(title, body):
    # cron has no DBUS_SESSION_BUS_ADDRESS; point at the user's session bus
    env = dict(os.environ)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS",
                   f"unix:path=/run/user/{os.getuid()}/bus")
    try:
        subprocess.run(["notify-send", "-u", "critical", title, body],
                       env=env, timeout=30)
    except Exception as e:
        print(f"notify-send failed: {e!r}")


def send_gmail_draft(draft_id, token_path):
    """Send an already-created draft. Uses users.drafts.send, which is
    authorized under the SAME gmail.compose scope used for create_draft
    (confirmed against Google's own API reference) — no new OAuth consent
    or token regeneration needed to enable this."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(
        token_path, scopes=["https://www.googleapis.com/auth/gmail.compose"]
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service.users().drafts().send(userId="me", body={"id": draft_id}).execute()


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest-dir", default="digests")
    ap.add_argument("--sheet-csv", required=True)
    ap.add_argument("--outline-txt", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--gmail-token", default="gmail_token.json")
    ap.add_argument("--mail-method", choices=["gmail", "smtp"], default="gmail",
                     help="'gmail' (default) uses the Gmail API draft/send flow. "
                          "'smtp' sends directly via an SMTP server (e.g. a local "
                          "exim4) — needs --smtp-host/--smtp-user, and the "
                          "SMTP_PASSWORD env var.")
    ap.add_argument("--smtp-host", default="localhost")
    ap.add_argument("--smtp-port", type=int, default=587)
    ap.add_argument("--smtp-user", default=None,
                     help="Required with --mail-method smtp")
    ap.add_argument("--smtp-from", default=None,
                     help="From address for SMTP send (defaults to --smtp-user)")
    ap.add_argument("--local-mail-to", default=getpass.getuser(),
                     help="Local recipient for the mail/mailx fallback when "
                          "SMTP is unreachable (defaults to current user)")
    ap.add_argument("--backend", choices=["api", "local"], default="api",
                     help="'api' uses ANTHROPIC_API_KEY (better instruction-"
                          "following, costs money). 'local' uses a GGUF model "
                          "via llama-cpp-python (free, needs --local-model, "
                          "weaker judgment quality).")
    ap.add_argument("--local-model", default=None,
                     help="Path to a GGUF file, 'auto', or a server name "
                          "(ollama, lmstudio, llama), or host:port[/model-id] "
                          "for llama-server (key from $LLAMA_API_KEY). 'auto' probes common "
                          "local servers then falls back to the default GGUF. "
                          "Required when --backend local (unless auto finds a server).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the email instead of creating a Gmail draft")
    ap.add_argument("--send", action="store_true",
                     help="After creating the draft, actually send it. Default "
                          "is draft-only (create_draft, no send) — this flag "
                          "is how you opt into unattended sending. Uses the "
                          "same gmail.compose-scoped token already set up; "
                          "no new OAuth consent needed.")
    args = ap.parse_args()

    if args.mail_method == "smtp":
        if not args.smtp_user:
            print("ERROR: --smtp-user is required with --mail-method smtp",
                  file=sys.stderr)
            sys.exit(1)
        if not os.environ.get("SMTP_PASSWORD"):
            print("ERROR: SMTP_PASSWORD env var is required with "
                  "--mail-method smtp", file=sys.stderr)
            sys.exit(1)

    if args.backend == "api":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set (needed for --backend api)",
                  file=sys.stderr)
            sys.exit(1)
    else:
        if not args.local_model:
            print("ERROR: --local-model is required with --backend local",
                  file=sys.stderr)
            sys.exit(1)
        if args.local_model == "auto":
            # Validate that auto will work — either a server is running or
            # the default GGUF exists. We do a quick probe here so the user
            # gets a clear error before processing starts.
            import importlib.util
            path = Path(__file__).parent / "ai-summary.py"
            spec = importlib.util.spec_from_file_location("_probe", str(path))
            _probe_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_probe_mod)
            detected = _probe_mod.detect_local_model()
            if not detected:
                default = _probe_mod.DEFAULT_GGUF_MODEL
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved = os.path.normpath(os.path.join(script_dir, default))
                if not os.path.exists(resolved):
                    print(
                        f"ERROR: --local-model auto found no running server "
                        f"and default GGUF not found at {resolved}",
                        file=sys.stderr)
                    sys.exit(1)
                print(f"[auto] No server detected, will use default GGUF: {default}")

    digest_path = latest_digest_file(args.digest_dir)
    if not digest_path:
        print(f"No digest files found in {args.digest_dir}/ — nothing to do")
        sys.exit(0)

    if already_processed(digest_path, args.digest_dir):
        print(f"{digest_path} was already emailed — nothing new. Exiting.")
        sys.exit(0)

    digest_text = Path(digest_path).read_text(encoding="utf-8")
    sheet_text = Path(args.sheet_csv).read_text(encoding="utf-8")
    outline_text = Path(args.outline_txt).read_text(encoding="utf-8")

    items = parse_digest_md(digest_text)
    past = parse_sheet_csv(sheet_text)
    topics = parse_outline_topics(outline_text)

    if not items:
        print(f"{digest_path} parsed to 0 items — nothing to do")
        sys.exit(0)

    candidates = flag_overlaps(items, past, topics)
    print(f"Parsed {len(items)} digest items, flagged overlaps against "
          f"{len(past)} past episodes + {len(topics)} outline topics.")

    past_titles = "; ".join(e["title"] for e in past)
    # Without this week's news the prompt is identical every run, and at
    # TEMP=0.01 the model returns the same idea week after week.
    news_titles = "; ".join(i["title"] for i in items)

    if args.backend == "local":
        ai_summary, model = load_local_model(args.local_model)
        idea = generate_idea_local(ai_summary, model, past_titles, news_titles)
        enriched = judge_candidates_local(candidates, ai_summary, model)
        kept = [e for e in enriched if e.get("decision") != "drop"]
        score_candidates_local(kept, ai_summary, model)  # attaches 'priority' in place
    else:
        idea = generate_idea_anthropic(past_titles, news_titles, api_key)
        enriched = judge_candidates(candidates, api_key)  # 'priority' already requested in the same call
        kept = [e for e in enriched if e.get("decision") != "drop"]

    print(f">> {idea}")
    if not idea:
        print("[WARNING] Idea generation returned empty — continuing without it.")
    print(f"Judgment ({args.backend}): {len(enriched)} evaluated, {len(kept)} kept.")

    week_of = Path(digest_path).stem.split("-hpc-digest")[0]
    subject, plain_body, html_body = render_email(enriched, idea, week_of=week_of)

    if args.dry_run:
        print("=== DRY RUN — not creating a draft ===")
        print(subject)
        print(plain_body)
        return

    # LLM work is done; free the GPU before possibly hours of send retries
    if int(os.environ.get("LLAMA_PID", 0)) > 0:
        try:
            os.kill(int(os.environ["LLAMA_PID"]), signal.SIGTERM)
            print("Stopped llama-server before sending")
        except ProcessLookupError:
            pass

    if args.mail_method == "smtp":
        if not args.send:
            print("--mail-method smtp has no draft mode — pass --send to "
                  "actually send it (nothing sent).")
        else:
            from_addr = args.smtp_from or args.smtp_user
            # SMTP host is sometimes unreachable for a while; retry before giving up
            delays = [5 * 60, 15 * 60, 30 * 60, 60 * 60]
            for attempt in range(len(delays) + 1):
                try:
                    send_smtp_email(
                        args.to, from_addr, subject, plain_body, html_body,
                        args.smtp_host, args.smtp_port, args.smtp_user,
                        os.environ["SMTP_PASSWORD"],
                    )
                    print(f"Sent via SMTP ({args.smtp_host}:{args.smtp_port}) to {args.to}")
                    break
                except (OSError, smtplib.SMTPException) as e:
                    print(f"SMTP send attempt {attempt + 1} failed: {e!r}")
                    if attempt < len(delays):
                        print(f"Retrying in {delays[attempt] // 60} min")
                        time.sleep(delays[attempt])
                        continue
                    unsent = Path(f"unsent-email-{week_of}.html").resolve()
                    unsent.write_text(html_body)
                    try:
                        used = send_local_mail(args.local_mail_to, subject, html_body)
                    except (OSError, subprocess.SubprocessError) as le:
                        print(f"Local mail fallback failed: {le!r}")
                        used = None
                    if not used:
                        notify_desktop(f"Digest email {week_of} NOT sent",
                                       f"{e!r}\nNo working mail/mailx. Saved to {unsent}")
                        raise
                    print(f"SMTP unreachable; sent via local `{used}` to {args.local_mail_to}")
    else:
        draft = create_gmail_draft(args.to, subject, plain_body, html_body, args.gmail_token)
        print(f"Created Gmail draft id={draft.get('id')}")

        if args.send:
            send_gmail_draft(draft["id"], args.gmail_token)
            print(f"Sent draft id={draft.get('id')} to {args.to}")
        else:
            print("Left as draft (pass --send to actually send it)")

    mark_processed(digest_path, args.digest_dir)


if __name__ == "__main__":
    main()

