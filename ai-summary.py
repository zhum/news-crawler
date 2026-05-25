#!/usr/bin/env python3

from langchain_ollama import ChatOllama                    # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from pathlib import Path
from datetime import date

import re
from rich.progress import Progress, MofNCompleteColumn, BarColumn
from rich.progress import TaskProgressColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn
import json
import sys
import yaml

MODEL = "qwen3:14b"
TEMP = 0.01
LINES = 300
BASE_DIR = "."

PROMPT_RATE_SUMMARY = """
You are an expert in High Performance Computing.
On a scale of 1 to 10, where 1 is "very uninteresting"
and 10 is "extremely interesting," how would you rate
the following text for an audience looking for news about new HPC
technologies, new and updated HPC software, benchmark results,
lower rate for new hardware installations?
Generate concise summary, 1-3 sentences, emphasize key point -
which technology, hardware or software is mentioned, what is the impact.
Print only rate and summary in json format:
'{{"rate": "RATE", "summary": "SUMMARY"}}'.
STRICTLY FOLLOW THIS FORMAT!

{article}
"""

PROMPT_TAG = (
    'On a scale of 1 to 10, how relevant is the topic "{tag}" ({desc}) '
    'to the following article? '
    'Answer only with a single integer number.\n\n{article}'
)


def _clean(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def generate_answer(file, lines=LINES, model=MODEL, temp=TEMP, tags=None):
    """Rate, summarise, and tag one article. tags is dict {name: description}."""
    with Path.open(file) as f:
        text = f.read()

    text_lines = text.splitlines()
    url = text_lines[0][5:]
    header = text_lines[1][8:]
    article = " ".join(text_lines[3:lines])

    chain = ChatOllama(
        model=model,
        temperature=temp) | StrOutputParser()  # type: ignore

    # Rate + summary
    response = chain.invoke(PROMPT_RATE_SUMMARY.format(article=article))
    try:
        data = json.loads(_clean(response))
    except Exception:
        data = {"rate": -1, "summary": response}

    # One rating call per tag
    tag_ratings = {}
    for tag_name, desc in (tags or {}).items():
        resp = chain.invoke(PROMPT_TAG.format(tag=tag_name, desc=desc, article=article))
        try:
            rating = int(re.search(r'\d+', _clean(resp)).group())
            rating = max(1, min(10, rating))
        except (AttributeError, ValueError):
            rating = 1
        tag_ratings[tag_name] = rating
    data['tags'] = tag_ratings

    data['header'] = header
    data['url'] = url
    return data


def archive_file(filepath, archive_dir):
    p = Path(filepath)
    if p.exists() and archive_dir:
        dest = Path(archive_dir) / f"{p.name}-{date.today()}"
        p.rename(dest)


def process_group(added_file, outfile, model=MODEL, temp=TEMP, lines=LINES,
                  archive_dir=None, tags=None):
    if outfile:
        archive_file(outfile, archive_dir)

    answers = []
    try:
        added = Path(added_file).read_text().splitlines()
        with Progress(
                      MofNCompleteColumn(),
                      BarColumn(),
                      TaskProgressColumn(),
                      TimeElapsedColumn(),
                      TimeRemainingColumn()
                     ) as progress:
            task = progress.add_task(
                f"Processing {added_file}", total=len(added))
            for file_line in added:
                _, file = file_line.split(' ')
                answer = generate_answer(f"{BASE_DIR}/{file}", lines, model, temp, tags)
                answer["file"] = str(file)
                answers.append(answer)
                progress.update(task, advance=1)
    except FileNotFoundError:
        print(f"File {added_file} is missing, skipping.")
        return

    if outfile:
        with open(outfile, 'w') as f:
            f.write(json.dumps(answers, indent=2))
    else:
        print(json.dumps(answers, indent=2))


def get_groups(config):
    if 'groups' in config:
        return config['groups']
    return [{
        'name': 'default',
        'sites': config.get('sites', []),
        'output': config.get('output', {}),
    }]


def process_config(config_path, archive_dir=None, group_filter=None):
    config = yaml.safe_load(Path(config_path).read_text())
    groups = get_groups(config)
    if group_filter:
        names = {n.strip() for n in group_filter.split(',')}
        groups = [g for g in groups if g.get('name') in names]

    ai_cfg = config.get('ai', {})
    default_model = ai_cfg.get('model', MODEL)
    default_temp = ai_cfg.get('temperature', TEMP)
    default_lines = ai_cfg.get('max_lines', LINES)
    all_tags = config.get('tags', {})

    for group in groups:
        # Per-group ai settings fall back to top-level ai section
        g_ai = group.get('ai', {})
        model = g_ai.get('model', default_model)
        temp = g_ai.get('temperature', default_temp)
        lines = g_ai.get('max_lines', default_lines)

        # Build tags dict for this group: {name: description}
        group_tag_keys = group.get('tags', list(all_tags.keys()))
        tags = {k: all_tags[k] for k in group_tag_keys if k in all_tags} or None

        out = group.get('output', {})
        added_file = out.get('added_file', 'added.txt')
        summary_file = out.get('summary_file', 'summary.json')
        print(f"\n=== Group: {group.get('name', 'default')} (model: {model}) ===")
        process_group(added_file, summary_file, model, temp, lines, archive_dir, tags)


# --- CLI entry point ---

if len(sys.argv) > 1 and sys.argv[1] == '--config':
    config_path = sys.argv[2] if len(sys.argv) > 2 else 'config.yaml'
    archive_dir = None
    group_filter = None
    args = sys.argv[3:]
    if '--archive' in args:
        idx = args.index('--archive')
        archive_dir = args[idx + 1]
    if '--group' in args:
        idx = args.index('--group')
        group_filter = args[idx + 1]
    process_config(config_path, archive_dir, group_filter)
else:
    outfile = sys.argv[1] if len(sys.argv) > 1 else None
    added_file = sys.argv[2] if len(sys.argv) > 2 else 'added.txt'
    process_group(added_file, outfile)
