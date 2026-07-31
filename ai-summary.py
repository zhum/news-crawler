#!/usr/bin/env python3
"""
Rate, summarise, and tag crawled articles using a local GGUF model via
llama-cpp-python.

The model is chosen at runtime (per group, from config), so the chat template
must match whatever family is loaded — feeding e.g. Gemma the Llama-3 control
tokens makes the model emit garbage. The CHAT_TEMPLATES / detect_model_family /
render_chat_prompt machinery below mirrors arxiv.py.
"""

from pathlib import Path
from datetime import date
from typing import List, Union

import re
from rich.progress import Progress, MofNCompleteColumn, BarColumn
from rich.progress import TaskProgressColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.console import Console
import json
import os
import sys
import yaml

from llama_cpp import Llama, LlamaGrammar

# Default model is now a GGUF file path (resolved relative to this script),
# not an Ollama model name.
MODEL = "../GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
TEMP = 0.01
LINES = 300
BASE_DIR = "."

CTX_LENGTH = 32768
MAX_NEW_TOKENS = 1500

# KV-cache quantization. ggml type codes: F16=1, Q8_0=8, Q4_0=2. Q4_0 quarters
# the KV cache vs f16 so it fits in VRAM. V-cache quant REQUIRES flash attention.
TYPE_K = 2  # Q4_0
TYPE_V = 2  # Q4_0

console = Console()

# Constrain the per-tag answer to a bare 1-2 digit integer. The grammar admits
# exactly one or two digits, so generation terminates on its own.
INT_GRAMMAR = r"""
    root ::= [0-9] [0-9]?
    """


# ══════════════════════════════════════════════════════════════════
# Prompt formatting — model-family-aware chat templates
# ══════════════════════════════════════════════════════════════════
CHAT_TEMPLATES = {
    "llama3": {
        "bos": "<|begin_of_text|>",
        "turn": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
        "gen": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_role": "assistant",
        "merge_system": False,
        "eot": "<|eot_id|>",
    },
    "chatml": {  # Qwen, and a safe modern default
        "bos": "",
        "turn": "<|im_start|>{role}\n{content}<|im_end|>\n",
        "gen": "<|im_start|>assistant\n",
        "assistant_role": "assistant",
        "merge_system": False,
        "eot": "<|im_end|>",
    },
    "gemma": {  # Gemma has no system role and requires user/model alternation
        "bos": "<bos>",
        "turn": "<start_of_turn>{role}\n{content}<end_of_turn>\n",
        "gen": "<start_of_turn>model\n",
        "assistant_role": "model",
        "merge_system": True,
        "eot": "<end_of_turn>",
    },
}


def detect_model_family(model_path: str) -> str:
    """Pick a chat-template family from the GGUF file name."""
    name = Path(model_path).name.lower()
    if "gemma" in name:
        return "gemma"
    if "qwen" in name or "qwq" in name:
        return "chatml"
    if "llama-3" in name or "llama3" in name or "llama-4" in name or "llama4" in name:
        return "llama3"
    return "chatml"  # safe default for unknown modern instruct models


def _normalize_messages(messages):
    """Return [(role, content)] from dicts, dropping empties."""
    out = []
    for m in messages:
        content = (m.get("content", "") or "").strip()
        role = str(m.get("role", "user")).split(".")[-1].lower()
        if content:
            out.append((role, content))
    return out


def _render_chat(messages, family: str) -> str:
    """Render a message list into the prompt string for the given family."""
    tmpl = CHAT_TEMPLATES[family]
    mapped, sys_buf = [], []
    for role, content in _normalize_messages(messages):
        if role == "system":
            if tmpl["merge_system"]:
                sys_buf.append(content)
                continue
            r = "system"
        elif role == "assistant":
            r = tmpl["assistant_role"]
        else:
            r = "user"
        mapped.append([r, content])

    # Families without a system role (Gemma): fold system into first user turn.
    if sys_buf:
        sys_text = "\n\n".join(sys_buf)
        for pair in mapped:
            if pair[0] == "user":
                pair[1] = f"{sys_text}\n\n{pair[1]}"
                break
        else:
            mapped.insert(0, ["user", sys_text])

    # Coalesce consecutive same-role turns (Gemma requires strict alternation).
    coalesced = []
    for r, c in mapped:
        if coalesced and coalesced[-1][0] == r:
            coalesced[-1][1] += "\n\n" + c
        else:
            coalesced.append([r, c])

    out = tmpl["bos"]
    for r, c in coalesced:
        out += tmpl["turn"].format(role=r, content=c)
    out += tmpl["gen"]
    return out


def render_chat_prompt(system: str, user: str, family: str,
                       assistant_prefix: str = "") -> str:
    """Build a single-turn (system + user) prompt and optionally prime the answer."""
    prompt = _render_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        family,
    )
    return prompt + assistant_prefix


class ModelWrapper:
    """Wrapper for a llama-cpp-python GGUF model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = CTX_LENGTH,
        n_gpu_layers: int = -1,
    ):
        # Resolve relative model paths against this script's directory so loading
        # works regardless of the current working directory. Absolute paths are
        # left unchanged by os.path.join.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(script_dir, model_path))
        self.model_path = model_path
        self.family = detect_model_family(model_path)
        console.print(f"[yellow]Loading GGUF model from {model_path}...[/yellow]")
        console.print(f"[yellow]Chat-template family: {self.family}[/yellow]")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,  # set 0 if you want pure CPU
                flash_attn=True,            # required for V-cache quantization
                type_k=TYPE_K,
                type_v=TYPE_V,
                verbose=False,
            )
            console.print("[green]Model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = TEMP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        grammar_str: Union[str, None] = None,
        stop: Union[List[str], None] = None,
    ) -> str:
        try:
            grammar = LlamaGrammar.from_string(grammar_str) if grammar_str else None

            # Stop only on the family's end-of-turn token by default, so JSON
            # spanning blank lines is not truncated. llama.cpp also stops on the
            # native EOS token.
            if stop is None:
                stop = [CHAT_TEMPLATES[self.family]["eot"]]

            result = self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.2,
                top_k=40,
                stop=stop,
                grammar=grammar,
                min_p=0.05,
            )
            return result["choices"][0]["text"].strip()  # type: ignore
        except Exception as e:
            console.print(f"[red]Error during generation: {e}[/red]")
            return ""


PROMPT_RATE_SUMMARY = """
You are an expert analyst covering the following topic: {topic}.
On a scale of 1 to 10, where 1 is "very uninteresting"
and 10 is "extremely interesting," how would you rate
the following text for an audience interested in {topic}?
Generate a concise summary in 1-3 sentences, emphasizing the key point.
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
    """Strip reasoning blocks (e.g. Qwen <think>...</think>)."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def _extract_json(response):
    """Pull the first {...} object out of a (possibly fenced) response."""
    text = _clean(response)
    text = re.sub(r'```(?:json)?', '', text).strip('` \n')
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


DEFAULT_TOPIC = "High Performance Computing: new technologies, software updates, benchmarks"


def generate_answer(file, model, lines=LINES, temp=TEMP, tags=None, topic=DEFAULT_TOPIC):
    """Rate, summarise, and tag one article. tags is dict {name: description}."""
    with Path.open(file) as f:
        text = f.read()

    text_lines = text.splitlines()
    url = text_lines[0][5:]
    header = text_lines[1][8:]
    article = " ".join(text_lines[3:lines])

    # Rate + summary
    prompt = render_chat_prompt(
        "", PROMPT_RATE_SUMMARY.format(article=article, topic=topic), model.family)
    response = model.generate(prompt, temperature=temp)
    try:
        data = json.loads(_extract_json(response))
    except Exception:
        data = {"rate": -1, "summary": _clean(response)}

    # One rating call per tag, grammar-constrained to a bare integer.
    tag_ratings = {}
    for tag_name, desc in (tags or {}).items():
        tag_prompt = render_chat_prompt(
            "", PROMPT_TAG.format(tag=tag_name, desc=desc, article=article),
            model.family)
        resp = model.generate(
            tag_prompt, temperature=temp, max_new_tokens=8,
            grammar_str=INT_GRAMMAR)
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
                  archive_dir=None, tags=None, topic=DEFAULT_TOPIC):
    if outfile:
        archive_file(outfile, archive_dir)

    # Load the GGUF model once and reuse it for every article in the group.
    try:
        model_wrapper = ModelWrapper(model)
    except Exception as e:
        console.print(f"[red]Failed to load model {model}: {e}[/red]")
        return

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
                answer = generate_answer(
                    f"{BASE_DIR}/{file}", model_wrapper, lines, temp, tags, topic)
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

        topic = group.get('topic', DEFAULT_TOPIC)
        out = group.get('output', {})
        added_file = out.get('added_file', 'added.txt')
        summary_file = out.get('summary_file', 'summary.json')
        print(f"\n=== Group: {group.get('name', 'default')} (model: {model}) ===")
        process_group(added_file, summary_file, model, temp, lines, archive_dir, tags, topic)


# --- CLI entry point ---
# Guarded so this file can also be imported as a module (e.g. by
# generate_digest_email.py, to reuse ModelWrapper/render_chat_prompt for
# local-model inference) without the CLI logic below firing on import.
if __name__ == "__main__":
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
