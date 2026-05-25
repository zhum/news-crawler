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

TAGS = '"hardware", "software", "benchmarks", "new technology", "investments", "market", "opensource", "new software", "software update", "supercomputing", "AI", "speculations", "other"'

PROMPT_1 = (
    """
You are an expert in High Performance Computing.
On a scale of 1 to 10, where 1 is "very uninteresting"
and 10 is "extremely interesting," how would you rate
the following text for a audience, looking for news about new HPC
technologies, new and updated HPC software, benchmark results,
lower rate for new hardware installations?
Add tags from set of {TAGS}, only if tags are really relevant.
Generate concise summary, 1-3 sentenses, emphasize key point -
which techology, hardware or software is pointed, what is impact.
Print only rate and sumary in json format:
'{{"rate": "RATE", "summary": "SUMMARY", "tags": "TAGSLIST"}}'.
STRICTLY FOLLOW THIS FORMAT!

{article}
"""
)


def build_tags_string(all_tags, group_tag_keys):
    """Build tag list string for the prompt from config tag definitions."""
    parts = []
    for key in group_tag_keys:
        desc = all_tags.get(key, '')
        parts.append(f'"{key}" ({desc})' if desc else f'"{key}"')
    return ', '.join(parts)


def generate_answer(file, lines=LINES, model=MODEL, temp=TEMP, tags=TAGS):
    """Generate an answer."""
    with Path.open(file) as f:
        text = f.read()

    text_lines = text.splitlines()
    url = text_lines[0][5:]
    header = text_lines[1][8:]
    article = " ".join(text_lines[3:lines])
    prompt1 = PROMPT_1.format(article=article, TAGS=tags)

    chain = ChatOllama(
        model=model,                           # type: ignore
        temperature=temp) | StrOutputParser()  # type: ignore
    response = chain.invoke(prompt1)
    try:
        clean_answer = re.sub(
            r'<think>.*</think>',
            '',
            response,
            flags=re.DOTALL)
        data = json.loads(clean_answer)
    except Exception:
        data = {"rate": -1, "summary": response}
    data["header"] = header
    data["url"] = url
    return data


def archive_file(filepath, archive_dir):
    p = Path(filepath)
    if p.exists() and archive_dir:
        dest = Path(archive_dir) / f"{p.name}-{date.today()}"
        p.rename(dest)


def process_group(added_file, outfile, model=MODEL, temp=TEMP, lines=LINES,
                  archive_dir=None, tags=TAGS):
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
    model = ai_cfg.get('model', MODEL)
    temp = ai_cfg.get('temperature', TEMP)
    lines = ai_cfg.get('max_lines', LINES)

    all_tags = config.get('tags', {})

    for group in groups:
        out = group.get('output', {})
        added_file = out.get('added_file', 'added.txt')
        summary_file = out.get('summary_file', 'summary.json')
        group_tag_keys = group.get('tags', list(all_tags.keys()))
        tags = build_tags_string(all_tags, group_tag_keys) if all_tags else TAGS
        print(f"\n=== Group: {group.get('name', 'default')} ===")
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
