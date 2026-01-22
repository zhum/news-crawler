#!/usr/bin/env python3

# from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama                    # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from pathlib import Path

# import os
import re
from rich.progress import Progress, MofNCompleteColumn, BarColumn
from rich.progress import TaskProgressColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn
import json
import sys

# MODEL = "ollama:gpt-oss:20b"
# MODEL = "magistral"
MODEL = "qwen3:14b"
# MODEL = "llama3.1-8b-long-ctx"
# MODEL = "gpt-oss-20b-long-ctx:latest"
#  MODEL = "qwen3:14b"
#  MODEL = "mistral-nemo"

TEMP = 0.01
LINES = 300  # 90
# BASE_DIR = os.environ.get('CRAWL_BASE', ".")
BASE_DIR = "."

TAGS = '''
"hardware", "software", "benchmarks",
"new technology", "investments", "market",
"opensource", "new software", "software update",
"supercomputing", "AI", "speculations", "other"
'''

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

PROMPT_2 = (
    """
You are an expert in High Performance Computing.
Generate a concise summary of the atricle below, using 1-3 sentences,
and print it in json format: '{{"summary": "SUMMARY"}}'

{article}
"""
)


def generate_answer(file):
    """Generate an answer."""
    with Path.open(file) as f:
        text = f.read()

    lines = text.splitlines()
    # header = "\n".join(lines[0:2])
    url = lines[0][5:]
    header = lines[1][8:]
    # lines[2] = "Text:\n"
    article = " ".join(lines[3:LINES])
    prompt1 = PROMPT_1.format(
        article=article,
        TAGS=TAGS,
    )
    # prompt2 = PROMPT_2.format(
    #     article=article
    # )

    chain = ChatOllama(
        model=MODEL,                           # type: ignore
        temperature=TEMP) | StrOutputParser()  # type: ignore
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
    # response += "\n==================\n"
    # response += chain.invoke(prompt2)
    # return f"{header}\n====\n{response}"
    return data


added_file = "added.txt"
outfile = None
if (len(sys.argv) > 1):
    outfile = sys.argv[1]
    if (len(sys.argv) > 2):
        added_file = sys.argv[2]

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
        task = progress.add_task("Processing files", total=len(added))
        for file_line in added:
            _, file = file_line.split(' ')
            answer = generate_answer(f"{BASE_DIR}/{file}")
            answer["file"] = str(file)
            # print(answer)
            answers += [answer]
            progress.update(task, advance=1)
except FileNotFoundError:
    print(f"File {added_file} is missing, nothing to process. Exiting.")
    exit(1)

if outfile:
    with open(outfile, 'w') as file:
        file.write(json.dumps(answers, indent=2))
else:
    print(json.dumps(answers, indent=2))
