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

import socket
import urllib.request
import urllib.error

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

LLAMA_SERVER_HOST = "localhost"
LLAMA_SERVER_PORT = 8011

# Common local model servers to probe when --local-model auto
_LOCAL_SERVERS = [
    # (name, host, port, health_endpoint)
    ("llama.cpp", "localhost", 8011, "/v1/models"),
    ("Ollama",   "localhost", 11434, "/api/tags"),
    ("LM Studio","localhost", 1234,  "/v1/models"),
    ("textgen",  "localhost", 5000,  "/v1/models"),
]

DEFAULT_GGUF_MODEL = "../GGUF/Qwen3-4B-UD-Q5_K_XL.gguf"

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
    # Qwen3.x hybrid-thinking models: chatml with an empty reasoning block
    # pre-filled (Qwen's official enable_thinking=False form). Without it the
    # model spends its whole token budget in <think> and never answers.
    "qwen3": {
        "bos": "",
        "turn": "<|im_start|>{role}\n{content}<|im_end|>\n",
        "gen": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
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
    if "qwen3" in name:
        return "qwen3"
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


class ServerModelWrapper:
    """Wrapper that talks to a running llama.cpp server via its OpenAI-compatible API."""

    def __init__(self, host: str = LLAMA_SERVER_HOST, port: int = LLAMA_SERVER_PORT):
        self.base_url = f"http://{host}:{port}/v1"
        self.family = "chatml"  # safe default — server already has the model loaded
        console.print(f"[yellow]Connecting to llama.cpp server at {self.base_url}...[/yellow]")
        # Quick health check
        try:
            req = urllib.request.Request(f"{self.base_url}/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data", [])
                if models:
                    model_id = models[0].get('id', '')
                    console.print(f"[green]Server running, model: {model_id or '?'}[/green]")
                    if model_id:
                        self.family = detect_model_family(model_id)
                else:
                    console.print("[yellow]Server running, but no models listed[/yellow]")
        except Exception as e:
            console.print(f"[red]Server health check failed: {e}[/red]")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = TEMP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        grammar_str: Union[str, None] = None,
        stop: Union[List[str], None] = None,
    ) -> str:
        # llama.cpp server does not support GBNF grammar via the /v1 API,
        # so we silently ignore grammar_str when talking to the server.
        #
        # The prompt is already a fully-rendered chat template (with control
        # tokens like <|start_header_id|>).  Sending it to /v1/chat/completions
        # would double-encode it (the server applies its own template), so we
        # use /v1/completions which treats the input as raw text instead.
        body = json.dumps({
            "model": "",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": 0.95,
            "repeat_penalty": 1.2,
            "top_k": 40,
            "stop": stop or [],
            "min_p": 0.05,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            # Long articles (~10k tokens) can take minutes of prompt
            # processing when the model is partly CPU-offloaded; a short
            # timeout just abandons work the server keeps doing anyway.
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["text"].strip()
        except Exception as e:
            console.print(f"[red]Error during generation: {e}[/red]")
            return ""


class ModelWrapper:
    """Wrapper for a llama-cpp-python GGUF model, or a remote llama.cpp server.

    model_path can be:
      - "auto"        — probe common local servers; fall back to default GGUF
      - "<path>"      — load GGUF from that path
      - "<server>"    — connect to a named server (e.g. "ollama", "lmstudio")
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = CTX_LENGTH,
        n_gpu_layers: int = -1,
    ):
        # ── "auto": detect running server, then fall back to default GGUF ──
        if model_path == "auto":
            detected = detect_local_model()
            if detected:
                backend, model_id = detected
                console.print(
                    f"[cyan]Detected local server: {backend} "
                    f"(model: {model_id or 'unknown'})[/cyan]")
                self._wrapped = ServerModelWrapper(
                    host=_LOCAL_SERVERS[0][1],  # llama.cpp default host
                    port=_LOCAL_SERVERS[0][2],
                )
                # Detect which server it was and adjust host/port
                for name, host, port, _ in _LOCAL_SERVERS:
                    if name == backend:
                        self._wrapped = ServerModelWrapper(host=host, port=port)
                        break
                self._detected = True
                return
            else:
                console.print(
                    "[yellow]No local server detected — falling back to "
                    "default GGUF model.[/yellow]")
                model_path = DEFAULT_GGUF_MODEL

        # ── Named server shortcut ──
        server_map = {
            "ollama":   ("localhost", 11434),
            "lmstudio": ("localhost", 1234),
            "textgen":  ("localhost", 5000),
            "llama":    ("localhost", 8011),
        }
        if model_path.lower() in server_map:
            host, port = server_map[model_path.lower()]
            console.print(
                f"[cyan]Connecting to {model_path} server at "
                f"{host}:{port}...[/cyan]")
            self._wrapped = ServerModelWrapper(host=host, port=port)
            self._detected = True
            return

        # ── Explicit host:port shortcut (any llama.cpp-server-compatible
        # endpoint on a non-default port, e.g. "localhost:8090") ──
        m = re.fullmatch(r"([\w.-]+):(\d+)", model_path)
        if m:
            host, port = m.group(1), int(m.group(2))
            console.print(
                f"[cyan]Connecting to server at {host}:{port}...[/cyan]")
            self._wrapped = ServerModelWrapper(host=host, port=port)
            self._detected = True
            return

        # ── Resolve relative model paths against this script's directory so
        # loading works regardless of the current working directory. ──
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(script_dir, model_path))
        self.model_path = model_path
        self._family = detect_model_family(model_path)
        console.print(f"[yellow]Loading GGUF model from {model_path}...[/yellow]")
        console.print(f"[yellow]Chat-template family: {self._family}[/yellow]")
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
        self._wrapped = self  # local mode: delegate to self
        self._detected = False

    @property
    def family(self) -> str:
        """Expose the chat-template family (local GGUF or server default)."""
        if self._wrapped is self:
            return self._family
        return self._wrapped.family  # type: ignore[attr-defined]

    def generate(
        self,
        prompt: str,
        temperature: float = TEMP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        grammar_str: Union[str, None] = None,
        stop: Union[List[str], None] = None,
    ) -> str:
        if self._wrapped is not self:
            # Remote server mode: delegate to ServerModelWrapper.
            return self._wrapped.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                grammar_str=grammar_str,
                stop=stop,
            )

        # Local GGUF mode: run inference directly via llama-cpp-python.
        try:
            grammar = LlamaGrammar.from_string(grammar_str) if grammar_str else None

            # Stop only on the family's end-of-turn token by default, so JSON
            # spanning blank lines is not truncated. llama.cpp also stops on the
            # native EOS token.
            if stop is None:
                stop = [CHAT_TEMPLATES[self._family]["eot"]]

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

# Article first, tag question last: every tag prompt for one article then
# shares a long identical prefix, which llama-server's prompt cache reuses
# instead of re-processing the whole article once per tag.
PROMPT_TAG = (
    '{article}\n\n'
    'On a scale of 1 to 10, how relevant is the topic "{tag}" ({desc}) '
    'to the article above? '
    'Answer only with a single integer number.'
)


def _clean(response):
    """Strip reasoning blocks (e.g. Qwen <think>...</think>)."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def _is_llama_server_running(host=LLAMA_SERVER_HOST, port=LLAMA_SERVER_PORT) -> bool:
    """Check whether a llama.cpp server is listening on the given host:port."""
    try:
        with socket.create_connection((host, port), timeout=2) as sock:
            sock.close()
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _probe_server(name, host, port, endpoint="/v1/models") -> bool:
    """Check if a local model server is alive and responding."""
    try:
        url = f"http://{host}:{port}{endpoint}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_local_model() -> tuple[str, str] | None:
    """Probe common local model servers. Returns (backend_type, model_id) or None."""
    for name, host, port, endpoint in _LOCAL_SERVERS:
        if _probe_server(name, host, port, endpoint):
            # Try to get the actual model name from the response
            try:
                url = f"http://{host}:{port}{endpoint}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode())
                    if name == "Ollama":
                        models = data.get("models", [])
                        model_id = models[0].get("name", "ollama") if models else "ollama"
                    else:
                        models = data.get("data", [])
                        model_id = models[0].get("id", name) if models else name
                    return (name, model_id)
            except Exception:
                return (name, "")
    return None


def start_default_model(model_path: str) -> None:
    """Print instructions for starting the default local model server."""
    console.print(f"\n[yellow]No local model server detected.[/yellow]")
    console.print(f"[yellow]Default GGUF model: {model_path}[/yellow]\n")
    console.print("[yellow]To use a local server, start one of these:[/yellow]")
    console.print("  1) llama.cpp:  llama-server -m {model_path} --port 8011 -c 32768")
    console.print("  2) Ollama:     ollama run qwen3:4b")
    console.print("  3) LM Studio:  (GUI — serve on port 1234)\n")
    console.print("[red]Aborting — no model available.[/red]")
    sys.exit(1)


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
            tag_prompt, temperature=temp, max_new_tokens=12,
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
    """Summarise one group's new articles into outfile. Returns False on
    failure (model unavailable), so the CLI can exit non-zero and crawl.sh
    stops instead of exporting last week's outfile as this week's digest."""
    # Load the GGUF model once and reuse it for every article in the group.
    try:
        model_wrapper = ModelWrapper(model)
    except Exception as e:
        console.print(f"[red]Failed to load model {model}: {e}[/red]")
        return False

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
        answers = []

    # Nothing new crawled: archive the previous outfile so export_digest.py
    # doesn't re-export last run's items as fresh news.
    if not answers:
        print(f"No new articles in {added_file}, archiving stale {outfile}.")
        if outfile:
            archive_file(outfile, archive_dir)
        return True

    if outfile:
        archive_file(outfile, archive_dir)
        with open(outfile, 'w') as f:
            f.write(json.dumps(answers, indent=2))
    else:
        print(json.dumps(answers, indent=2))
    return True


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

    ok = True
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
        ok &= process_group(added_file, summary_file, model, temp, lines,
                            archive_dir, tags, topic)
    return ok


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
        ok = process_config(config_path, archive_dir, group_filter)
    else:
        outfile = sys.argv[1] if len(sys.argv) > 1 else None
        added_file = sys.argv[2] if len(sys.argv) > 2 else 'added.txt'
        ok = process_group(added_file, outfile)
    sys.exit(0 if ok else 1)
