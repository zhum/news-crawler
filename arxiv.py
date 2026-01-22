#!/usr/bin/env python3
"""
ArXiv article parser and summarizer.
Reads arxiv digest files, splits them into individual articles,
and generates summaries using transformers library with GGUF models.

Llama-3.2-3B-Instruct-Q8_0.gguf from bartowski/Llama-3.2-3B-Instruct-GGUF
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from rich.progress import Progress, MofNCompleteColumn, BarColumn
from rich.progress import TaskProgressColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.console import Console

from llama_cpp import Llama, LlamaGrammar  # Add LlamaGrammar to imports
from typing import Union

ABSTRACT_HEAD = 700
CTX_LENGTH = 32768
MAX_NEW_TOKENS = 1500

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     import torch
# except ImportError:
#     AutoTokenizer = None
#     AutoModelForCausalLM = None
#     torch = None

console = Console()
JSON_GRAMMAR = r"""
    root   ::= object
    value  ::= number | string
    object ::= "{" space ( pair ( "," space pair )* )? "}" space
    pair   ::= string ":" space number
    string ::= "\"" [^"]* "\""
    number ::= [0-9]+
    space  ::= " "*
    """


class ModelWrapper:
    """Wrapper for llama-cpp-python GGUF model."""
    def __init__(
        self,
        model_path: str,
        n_ctx: int = CTX_LENGTH,
        n_gpu_layers: int = -1,
    ):
        self.model_path = model_path
        console.print(
            f"[yellow]Loading GGUF model from {model_path}...[/yellow]")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,  # set 0 if you want pure CPU
                verbose=False,
            )
            console.print("[green]Model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = MAX_NEW_TOKENS,
        grammar_str: Union[str, None] = None,
        stop: Union[List[str], None] = None,
        do_sample: bool = True,
    ) -> str:
        # llama-cpp uses max_tokens, not max_new_tokens
        try:
            if grammar_str:
                grammar = LlamaGrammar.from_string(grammar_str)
            else:
                grammar = None

            if stop is None:
                stop = [
                    "<|eot_id|>",
                    "<|end_of_text|>",
                    "\n\n",
                    "Note:"
                ]

            result = self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.2,
                top_k=40,
                stop=stop,
                grammar=grammar,
                min_p=0.05,      # Great for Llama 3.1
            )
            text = result["choices"][0]["text"]  # type: ignore
            return text.strip()
        except Exception as e:
            console.print(f"[red]Error during generation: {e}[/red]")
            return ""


class ArxivArticle:
    """Represents a single arxiv article."""

    def __init__(self, raw_text: str):
        self.raw_text = raw_text.strip()
        self.arxiv_id = self._extract_arxiv_id()
        self.title = self._extract_title()
        self.authors = self._extract_authors()
        self.categories = self._extract_categories()
        self.abstract = self._extract_abstract()
        self.url = self._extract_url()
        self.date = self._extract_date()

    def _extract_arxiv_id(self) -> Optional[str]:
        """Extract arxiv ID from the article text."""
        match = re.search(r'arXiv:(\d+\.\d+)', self.raw_text)
        return match.group(1) if match else None

    def _extract_title(self) -> Optional[str]:
        """Extract title from the article text."""
        match = re.search(
            r'Title:\s*(.+?)(?:\nAuthors:|$)', self.raw_text, re.DOTALL)
        if match:
            # Clean up title - remove extra whitespace and newlines
            title = ' '.join(match.group(1).split())
            return title
        return None

    def _extract_authors(self) -> Optional[str]:
        """Extract authors from the article text."""
        match = re.search(
            r'Authors:\s*(.+?)(?:\nCategories:|$)', self.raw_text, re.DOTALL)
        if match:
            authors = ' '.join(match.group(1).split())
            return authors
        return None

    def _extract_categories(self) -> Optional[str]:
        """Extract categories from the article text."""
        match = re.search(r'Categories:\s*(.+?)(?:\n|$)', self.raw_text)
        return match.group(1).strip() if match else None

    def _extract_abstract(self) -> Optional[str]:
        """Extract abstract from the article text."""
        # Abstract is between \\ markers
        match = re.search(r'\\\\\n\s+(.+?)\n\\\\', self.raw_text, re.DOTALL)
        if match:
            # Clean up abstract - normalize whitespace
            abstract = ' '.join(match.group(1).split())
            return abstract
        return None

    def _extract_url(self) -> Optional[str]:
        """Extract URL from the article text."""
        match = re.search(r'https://arxiv\.org/abs/[\d.]+', self.raw_text)
        return match.group(0) if match else None

    def _extract_date(self) -> Optional[str]:
        """Extract submission date from the article text."""
        match = re.search(r'Date:\s*(.+?)\s+GMT', self.raw_text)
        return match.group(1).strip() if match else None

    def is_valid(self) -> bool:
        """Check if article has minimum required fields."""
        return bool(self.arxiv_id and self.title and self.abstract)

    def to_dict(self) -> Dict:
        """Convert article to dictionary."""
        return {
            'arxiv_id': self.arxiv_id,
            'title': self.title,
            'authors': self.authors,
            'categories': self.categories,
            'abstract': self.abstract,
            'url': self.url,
            'date': self.date,
        }

    def __str__(self) -> str:
        """String representation for display."""
        return f"arXiv:{self.arxiv_id} - {self.title}"


def split_articles(content: str) -> List[ArxivArticle]:
    """
    Split the arxiv digest content into individual articles.

    Articles are separated by lines of dashes and start with arXiv: prefix.
    """
    articles = []

    # Split by the separator pattern (multiple dashes)
    parts = re.split(r'-{10,}\n', content)

    for part in parts:
        part = part.strip()
        if not part or not part.startswith('\\'):
            continue

        # Check if this part contains an arxiv ID
        if 'arXiv:' in part:
            article = ArxivArticle(part)
            if article.is_valid():
                articles.append(article)

    return articles


# class OldModelWrapper:
#     """Wrapper for transformers model with GGUF support."""

#     def __init__(self, model_path: str, device: str = "auto"):
#         """
#         Initialize the model wrapper.

#         Args:
#             model_path: Path to GGUF model file or HuggingFace model name
#             device: Device to use ('auto', 'cuda', 'cpu')
#         """
#         if AutoTokenizer is None or AutoModelForCausalLM is None:
#             raise ImportError(
#                 "transformers library not installed. "
#                 "Install with: pip install transformers[torch]"
#             )

#         self.model_path = model_path
#         self.device = device

#         console.print(f"[yellow]Loading model from {model_path}...[/yellow]")

#         try:
#             # Load tokenizer and model
#             # For GGUF files, use gguf_file parameter
#             # Note: GGUF support requires transformers >= 4.37.0
#             if model_path.endswith('.gguf'):
#                 # model_name = model_path[0:-5]
#                 model_name = "unsloth/Qwen3-4B-GGUF"
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     model_name,
#                     gguf_file=model_path,
#                 )
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_name,
#                     gguf_file=model_path,
#                     device_map=device,
#                 )
#             else:
#                 # Regular HuggingFace model
#                 self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_path,
#                     device_map=device,
#                     # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 )

#             console.print("[green]Model loaded successfully[/green]")

#         except Exception as e:
#             console.print(f"[red]Error loading model: {e}[/red]")
#             raise

#     def generate(
#         self,
#         prompt: str,
#         temperature: float = 0.1,
#         max_new_tokens: int = 500,
#         do_sample: bool = True,
#     ) -> str:
#         """
#         Generate text from a prompt.

#         Args:
#             prompt: Input prompt
#             temperature: Sampling temperature
#             max_new_tokens: Maximum new tokens to generate
#             do_sample: Whether to use sampling

#         Returns:
#             Generated text
#         """
#         try:
#             # Tokenize input
#             inputs = self.tokenizer(prompt, return_tensors="pt")

#             # Move to same device as model
#             if torch.cuda.is_available() and self.device != "cpu":
#                 inputs = {
#                     k: v.to(self.model.device)
#                     for k, v in inputs.items()
#                     }

#             # Generate
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     do_sample=do_sample,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                 )

#             # Decode output
#             generated_text = self.tokenizer.decode(
#                 outputs[0][inputs['input_ids'].shape[1]:],
#                 skip_special_tokens=True
#             )

#             return generated_text.strip()

#         except Exception as e:
#             console.print(f"[red]Error during generation: {e}[/red]")
#             return ""


def summarize_with_transformers(
    article: ArxivArticle,
    model: ModelWrapper,
    prompt_template: str,
    temperature: float = 0.1,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Optional[str]:
    """
    Generate a summary using transformers library.

    Args:
        article: ArxivArticle object to summarize
        model: ModelWrapper instance
        prompt_template:
        Template string with {title}, {abstract}, {categories} placeholders
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated summary text or None if failed
    """
    # Format the prompt
    prompt = prompt_template.format(
        title=article.title or "",
        abstract=article.abstract or "",
        categories=article.categories or "",
        authors=article.authors or "",
    )

    try:
        summary = model.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return summary

    except Exception as e:
        console.print(
            f"[red]Error generating summary for {article.arxiv_id}: {e}[/red]")
        return None


def process_files(
    file_paths: List[Path],
    model_path: str,
    output_file: Optional[Path] = None,
    prompt_template: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = MAX_NEW_TOKENS,
) -> List[Dict]:
    """
    Process multiple arxiv digest files and generate summaries.

    Args:
        file_paths: List of file paths to process
        model_path: Path to GGUF model file or HuggingFace model name
        output_file: Optional output JSON file path
        prompt_template: Custom prompt template
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of article dictionaries with summaries
    """
    if prompt_template is None:
        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in computer science research. Summarize the paper in 2-3 concise, technical sentences.
DO NOT provide notes, suggestions, or conversational filler. Output ONLY the summary.<|eot_id|><|start_header_id|>user<|end_header_id|>
Title: {title}
Categories: {categories}
Abstract: {abstract}

Provide the summary now.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
SUMMARY: """

    all_results = []
    total_articles = 0

    # First pass: count articles
    for file_path in file_paths:
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            continue

        content = file_path.read_text(encoding='utf-8', errors='ignore')
        articles = split_articles(content)
        total_articles += len(articles)

    console.print(
        f"[green]Found {total_articles} articles across"
        f" {len(file_paths)} file(s)[/green]")

    # Load model once for all articles
    try:
        model = ModelWrapper(model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return []

    # Second pass: process articles with progress bar
    with Progress(
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Processing articles", total=total_articles)

        for file_path in file_paths:
            if not file_path.exists():
                continue

            content = file_path.read_text(encoding='utf-8', errors='ignore')
            articles = split_articles(content)

            for article in articles:
                result = article.to_dict()

                # Generate summary using transformers
                summary = summarize_with_transformers(
                    article,
                    model,
                    prompt_template,
                    temperature,
                    max_tokens,
                )

                result['summary'] = summary
                console.print(f": {summary}")
                console.print("----------------")
                result['source_file'] = str(file_path)
                all_results.append(result)

                progress.update(task, advance=1)

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Results saved to {output_file}[/green]")

    return all_results


def post_process(model_path: str):
    """
    Post-process arxiv summaries to add topic rankings.

    Analyzes each article summary and ranks it (1-10)
    on multiple research areas.
    """
    RANKING_AREAS = [
        "New AI Model",
        "New Hardware Technology",
        "New software library, driver, framework",
        "Parallel computing",
        "Hardware Architecture",
        "Computational Engineering",
        "Active Sensing & Perception",
        "Autonomous Systems",
        "Decision-Theoretic Modeling",
        "Job Scheduling & Orchestration",
        "Fault Tolerance & Checkpointing",
        "Interconnects & Networking",
        "Energy & Power Awareness",
        "GPU/TPU/NPU/LPU Acceleration",
        "Computational Engineering",
        "Agentic Reasoning",
        "Ethical AI & Alignment",
        "Adversarial Training",
        "Scaling Trends & MoE",
        "Commonsense Reasoning",
        "Open Source"
    ]

    NEW_PROMPT = """Analyze the following research paper and rate it as new or updated result on a scale of 1-10, where:
- 1 = No new results, just a didgest of previous results
- 5 = Update of previous results
- 10 = Completely new result/software/hardware/idea.

Title: {title}
Abstract: {abstract}
Summary: {summary}

Return ONLY final JSON, nothing else.
Result:
{{"New Rate": X}}"""

    RANKING_PROMPT = """Analyze the following research paper and rate its relevance to EACH research area on a scale of 1-10, where:
- 1 = Not relevant at all
- 5 = Somewhat relevant
- 10 = Highly relevant and central to the area

Rate the paper on these research areas:
{areas}

Title: {title}
Abstract: {abstract}
Summary: {summary}

Return ONLY final JSON, nothing else.
Result:
{{{json_sample}}}"""  # type: ignore

    console.print("[yellow]Loading arxiv_summaries.json...[/yellow]")

    with open("arxiv_summaries.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    console.print(f"[green]Found {len(articles)} articles to analyze[/green]")

    # Load model
    console.print("[yellow]Loading model for ranking analysis...[/yellow]")
    try:
        model = ModelWrapper(model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return 1

    # Process each article
    with Progress(
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Analyzing articles", total=len(articles))

        middle = int(len(RANKING_AREAS)/2)
        for article in articles:
            # Format the ranking prompt
            areas_list1 = "\n".join(
                [f"- {area}" for area in RANKING_AREAS[:middle]]
                )
            prompt1 = RANKING_PROMPT.format(
                title=article.get('title', ''),
                abstract=article.get('abstract', '')[:ABSTRACT_HEAD],
                # ^ Limit abstract length
                summary=article.get('summary', ''),
                areas=areas_list1,
                json_sample=",".join(list([
                    f"\"{x}\": X" for x in RANKING_AREAS[:middle]
                ]))
            )

            areas_list2 = "\n".join(
                [f"- {area}" for area in RANKING_AREAS[middle:]]
                )
            prompt2 = RANKING_PROMPT.format(
                title=article.get('title', ''),
                abstract=article.get('abstract', '')[:ABSTRACT_HEAD],
                # ^ Limit abstract length
                summary=article.get('summary', ''),
                areas=areas_list2,
                json_sample=",".join(list([
                    f"\"{x}\": X" for x in RANKING_AREAS[middle:]
                ]))
            )
            prompt_new = NEW_PROMPT.format(
                title=article.get('title', ''),
                abstract=article.get('abstract', '')[:ABSTRACT_HEAD],
                # ^ Limit abstract length
                summary=article.get('summary', ''),
                )

            try:
                article['rankings'] = {}
                for prompt in (prompt1, prompt2, prompt_new):
                    # Generate rankings
                    response = model.generate(
                        prompt=prompt,
                        temperature=0.1,
                        max_new_tokens=MAX_NEW_TOKENS,
                        grammar_str=JSON_GRAMMAR
                    )

                    # Parse JSON response
                    # Clean up response - remove markdown
                    # code blocks if present
                    response = response.strip()
                    if response.startswith('```'):
                        # Remove markdown code block markers
                        lines = response.split('\n')
                        response = '\n'.join(
                            [line for line in lines
                             if not line.startswith('```')])
                        response = response.strip()

                    rankings = json.loads(response)
                    article['rankings'].update(rankings)

                console.print(
                    f"  [cyan]{article['arxiv_id']}[/cyan]: Ranked successfully")

            except json.JSONDecodeError as e:
                console.print(
                    "[yellow]Failed to parse rankings for "
                    f"{article.get('arxiv_id')}: {e}[/yellow]")
                console.print(
                    f"[yellow]Response was: {response}[/yellow]")
                # Set default rankings
                article['rankings'] = {area: 1 for area in RANKING_AREAS}
            except Exception as e:
                console.print(
                    f"[red]Error processing {article.get('arxiv_id')}: "
                    f"{e}[/red]")
                article['rankings'] = {area: 1 for area in RANKING_AREAS}

            progress.update(task, advance=1)

    # Save updated results
    output_file = "arxiv_summaries_ranked.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Ranked summaries saved to {output_file}[/green]")

    # Display top articles per category
    console.print("\n[bold]Top articles per category:[/bold]\n")
    for area in RANKING_AREAS:
        # Sort articles by this area's ranking
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get('rankings', {}).get(area, 0),
            reverse=True
        )

        top_article = sorted_articles[0]
        top_score = top_article.get('rankings', {}).get(area, 0)

        if top_score >= 7:  # Only show if highly relevant
            console.print(f"[cyan]{area}[/cyan] (score: {top_score}):")
            console.print(
                f"  {top_article['arxiv_id']}: {top_article['title'][:80]}...")
            console.print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Parse arxiv digest files and generate'
        ' summaries using transformers library'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Arxiv digest file(s) to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to GGUF model file or HuggingFace model'
        ' name (required unless --dry-run)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default='arxiv_summaries.json',
        help='Output JSON file (default: arxiv_summaries.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Custom prompt template (use {title}, {abstract},'
        ' {categories}, {authors})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens to generate (default: 500)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse files without generating summaries'
    )
    parser.add_argument(
        '--post-process',
        action='store_true',
        help='Analyze arxiv_summaries.json and add topic '
             'rankings (requires --model)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.post_process and not args.model:
        parser.error(
            "--model is required unless --dry-run or "
            "--post-process is specified")

    # Dry run mode - just parse and display
    if args.dry_run:
        for file_path in args.files:
            if not file_path.exists():
                console.print(f"[red]File not found: {file_path}[/red]")
                continue

            content = file_path.read_text(encoding='utf-8', errors='ignore')
            articles = split_articles(content)

            console.print(f"\n[bold]File: {file_path}[/bold]")
            console.print(f"Found {len(articles)} articles:\n")

            for i, article in enumerate(articles, 1):
                console.print(f"{i}. {article}")
                console.print(f"   Categories: {article.categories}")
                console.print(f"   URL: {article.url}\n")

        return 0

    if args.post_process:
        if not args.model:
            parser.error("--model is required for --post-process")
        return post_process(args.model)

    # Full processing mode
    results = process_files(
        args.files,
        args.model,
        args.output,
        args.prompt,
        args.temperature,
        args.max_tokens,
    )

    console.print("\n[green]Successfully processed "
                  f"{len(results)} articles[/green]")

    # Display sample results
    if results:
        console.print("\n[bold]Sample results:[/bold]\n")
        for result in results[:3]:
            console.print(
                f"[cyan]{result['arxiv_id']}[/cyan]: {result['title']}")
            if result.get('summary'):
                console.print(f"  Summary: {result['summary'][:200]}...")
            console.print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
