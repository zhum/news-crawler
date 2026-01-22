# HPC News Crawler & AI Summarizer

A Python-based web crawler and AI-powered summarization system for High Performance Computing (HPC) news and articles. This project automatically scrapes HPC-related news from multiple sources, generates AI summaries, and provides a web interface for browsing the content.

## Project Overview

This tool crawls HPC news websites, extracts article content, generates AI-powered summaries and ratings, and presents the results through a simple web interface. It's designed for HPC enthusiasts who want to stay updated with the latest developments in the field.

## Features

- Automated web scraping from multiple HPC news sources
- RSS feed support
- AI-powered article summarization using Ollama/LLM
- Relevance rating system (1-10 scale)
- Tag generation for categorization
- Web-based viewer with sorting capabilities
- Markdown conversion of articles
- Duplicate detection to avoid re-crawling
- ArXiv digest parser with llama.cpp summarization support

## Project Structure

```text
crawler/
├── config.yaml              # Main configuration file
├── pyproject.toml          # Python project dependencies and settings
├── scrapy.cfg              # Scrapy framework configuration
├── crawl.sh                # Main crawler execution script
├── ai-summary.py           # AI summarization script (Ollama)
├── arxiv.py                # ArXiv digest parser & summarizer (llama.cpp)
├── web_server.py           # HTTP server for viewing summaries
├── web-server.sh           # Web server launcher script
├── newscrawler/            # Scrapy spider package
│   ├── __init__.py
│   ├── items.py            # Scrapy item definitions
│   ├── middlewares.py      # Scrapy middlewares
│   ├── pipelines.py        # Scrapy pipelines
│   ├── settings.py         # Scrapy settings
│   └── spiders/
│       └── news_spider.py  # Main news spider implementation
├── texts/                  # Crawled article storage (created at runtime)
├── arch/                   # Archive directory for old summaries
├── added.txt               # Log of newly added articles
├── summary.json            # Generated AI summaries
└── newscrawler.log         # Application log file
```

## Key Files

### Configuration Files

#### [config.yaml](config.yaml)

Main configuration file containing:

- **spider**: Web crawler settings (user agent, delays, robots.txt compliance)
- **logging**: Log file path and level configuration
- **ai**: AI model settings (model name, temperature, context limits)
- **sites**: Array of news sources with CSS selectors:
  - `name`: Site identifier
  - `url`: Target URL or RSS feed
  - `css`: CSS selector for article list
  - `link`: CSS selector for article links
  - `headercss`: CSS selector for article titles (on article page)
  - `articlecss`: CSS selector for article content (on article page)
- **output**: Output directory configuration

#### [pyproject.toml](pyproject.toml)

Poetry-based Python project configuration with dependencies:

- **Core**: scrapy, markdownify, langchain-ollama, pyyaml, feedparser, rich
- **Dev tools**: pytest, black, isort, flake8, mypy
- Code formatting settings (Black, isort)
- Type checking configuration (mypy)

#### [scrapy.cfg](scrapy.cfg)

Scrapy framework configuration pointing to `newscrawler.settings` module.

### Core Scripts

#### [crawl.sh](crawl.sh)

Main execution script that:

1. Archives previous results (added.txt, summary.json)
2. Activates Python virtual environment
3. Runs the Scrapy spider (optionally filtered by site)
4. Executes AI summarization on results
5. Supports optional site filtering: `./crawl.sh "site_name"`

#### [ai-summary.py](ai-summary.py)

AI-powered summarization engine that:

- Reads newly added articles from `added.txt`
- Processes article content (first 300 lines by default)
- Uses Ollama LLM (default: gpt-oss-20b-long-ctx) to generate:
  - Relevance rating (1-10 scale)
  - Concise summary (1-3 sentences)
  - Categorization tags
- Outputs structured JSON with article metadata
- Features progress bar using Rich library

Key configuration:

- `MODEL`: LLM model name (default: "gpt-oss-20b-long-ctx:latest")
- `TEMP`: Temperature setting for generation (0.01 for consistency)
- `LINES`: Number of lines to process per article (300)

#### [web_server.py](web_server.py)

Simple HTTP server (port 8000) providing:

- Web interface for browsing summaries
- Sortable table by title, URL, or rating
- Color-coded ratings (green/yellow/red)
- Responsive HTML with clean CSS
- Direct links to original articles
- Article statistics display

Access at: `http://localhost:8000`

### Spider Implementation

#### [newscrawler/spiders/news_spider.py](newscrawler/spiders/news_spider.py)

Main spider class (`HpcwireSpider`) that:

- Loads configuration from `config.yaml`
- Supports site filtering via `only` parameter
- Handles both regular HTML and RSS feeds
- Extracts article content using configurable CSS selectors
- Converts HTML to Markdown
- Saves articles as text files with metadata
- Tracks new articles in `added.txt`
- Implements duplicate detection (skips existing files)
- Supports two modes:
  - Standard: Crawls article lists and follows links
  - One-page: Extracts multiple articles from single page

Article file format:

```text
URL: [article_url]
Header: [article_title]
Text:
[markdown_content]
```

## Workflow

1. **Crawling Phase** ([crawl.sh](crawl.sh))
   - Archives previous run data
   - Launches Scrapy spider
   - Spider visits configured news sources
   - Extracts articles not previously crawled
   - Converts HTML to Markdown
   - Saves to `texts/` directory
   - Logs new files to `added.txt`

2. **AI Summarization Phase** ([ai-summary.py](ai-summary.py))
   - Reads list of new articles from `added.txt`
   - Processes each article through LLM
   - Generates rating, summary, and tags
   - Outputs to `summary.json`

3. **Viewing Phase** ([web_server.py](web_server.py))
   - Serves HTML interface
   - Loads `summary.json`
   - Provides sorting and filtering
   - Links to original articles

## Setup

### Prerequisites

- Python 3.12+
- Ollama with compatible LLM model installed
- Virtual environment support

### Installation

1. Install dependencies using Poetry:

```bash
poetry install
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Ensure Ollama is running with the configured model:

```bash
ollama pull gpt-oss-20b-long-ctx
```

1. Make scripts executable:

```bash
chmod +x crawl.sh ai-summary.py web_server.py
```

## Usage

### Run Full Crawl and Summarization

```bash
./crawl.sh
```

### Crawl Specific Site(s)

```bash
./crawl.sh "insidehpc"
./crawl.sh "insidehpc,hpcwire academia"
```

### Run Summarization Only

```bash
./ai-summary.py summary.json
```

### Launch Web Viewer

```bash
./web_server.py
```

Then open: `http://localhost:8000`

### Run Spider Directly

```bash
scrapy crawl news
scrapy crawl news -a only="insidehpc"
```

## Configuration

### Adding New News Sources

Edit [config.yaml](config.yaml) and add to the `sites` array:

```yaml
- name: "your_site_name"
  url: "https://example.com/news"
  css: ".article-list-item"
  link: "a.article-link::attr(href)"
  headercss: "h1.title::text"
  articlecss: "div.content"
```

For RSS feeds:

```yaml
- name: "rss_site"
  url: "https://example.com/feed.rss"
  css: rss
  link: link
  headercss: "h1::text"
  articlecss: "article.content"
```

### Adjusting AI Model

Edit [ai-summary.py](ai-summary.py):

```python
MODEL = "your-model-name"
TEMP = 0.01  # Increase for more creative summaries
LINES = 300  # Number of lines to analyze
```

### Crawler Behavior

Edit spider settings in [config.yaml](config.yaml):

```yaml
spider:
  user_agent: "newsfetcher (HPC enthusiast)"
  download_delay: 1  # Seconds between requests
  concurrent_requests_per_domain: 1
  robotstxt_obey: true
```

## Output Files

- `texts/`: Directory containing crawled articles as markdown text files
- `added.txt`: Log of newly added articles with timestamps
- `summary.json`: AI-generated summaries with ratings and metadata
- `newscrawler.log`: Application logs
- `arch/`: Archive directory for historical data

## Summary JSON Format

```json
[
  {
    "rate": "8",
    "summary": "Brief article summary here.",
    "tags": "software, benchmarks",
    "header": "Article Title",
    "url": "https://example.com/article",
    "file": "texts/example-com-article.txt"
  }
]
```

## ArXiv Digest Parser

The [arxiv.py](arxiv.py) script provides a standalone tool for parsing ArXiv email digests and generating summaries using the transformers library with GGUF model support.

### Features

- Parses ArXiv email digest files
- Extracts individual articles with metadata (title, authors, categories, abstract, URL)
- Generates summaries using transformers library with GGUF models or HuggingFace models
- GPU acceleration support (CUDA)
- Supports batch processing of multiple files
- Rich terminal output with progress bars
- Exports results to JSON

### Usage

#### Dry Run (Parse Only)

Test the parser without generating summaries:

```bash
./arxiv.py txt --dry-run
```

This will display all parsed articles with their metadata.

#### Generate Summaries

Process ArXiv digest files and generate summaries:

```bash
./arxiv.py txt --model /path/to/model.gguf --output arxiv_summaries.json
```

#### Multiple Files

Process multiple digest files at once:

```bash
./arxiv.py file1.txt file2.txt file3.txt --model /path/to/model.gguf
```

#### Custom Settings

```bash
./arxiv.py txt \
  --model /path/to/model.gguf \
  --device cuda \
  --temperature 0.2 \
  --max-tokens 300 \
  --output my_summaries.json
```

#### Using HuggingFace Models

Instead of GGUF files, you can use HuggingFace model names:

```bash
./arxiv.py txt --model microsoft/phi-2 --device cuda
```

#### Custom Prompt Template

```bash
./arxiv.py txt --model model.gguf \
  --prompt "Summarize this paper briefly: {title}. Abstract: {abstract}. Summary:"
```

Available template variables: `{title}`, `{abstract}`, `{categories}`, `{authors}`

### Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "arxiv_id": "2601.04214",
    "title": "Active Sensing Shapes Real-World Decision-Making...",
    "authors": "Hongliang Lu, Yunmeng Liu, Junjie Yang",
    "categories": "cs.AI cs.HC cs.RO q-bio.NC",
    "abstract": "Human decision-making heavily relies on active sensing...",
    "url": "https://arxiv.org/abs/2601.04214",
    "date": "Thu, 18 Dec 2025 05:45:53",
    "summary": "This paper presents a computational scheme that generalizes...",
    "source_file": "txt"
  }
]
```

### Requirements

- Python 3.12+
- transformers library
- torch (PyTorch)
- rich library (for terminal output)
- GGUF model file or HuggingFace model name
- Optional: CUDA for GPU acceleration
- Note: if using cuda reinstall llama-cpp-python! `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir`

Install dependencies:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir
pip install transformers torch rich
```

### Example Workflow

1. Receive ArXiv digest email
1. Save to text file
1. Parse and summarize:

```bash
# With GGUF model
./arxiv.py arxiv_digest.txt --model qwen2.5-7b.gguf --device cuda -o summaries.json

# With HuggingFace model
./arxiv.py arxiv_digest.txt --model microsoft/phi-2 --device cuda -o summaries.json
```

1. Review summaries in JSON file

## Development

### Code Quality Tools

Configured in [pyproject.toml](pyproject.toml):

- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

### Run Quality Checks

```bash
black .
isort .
flake8 .
mypy .
pytest
```

## Dependencies

### Core Dependencies

- **scrapy**: Web crawling framework
- **markdownify**: HTML to Markdown conversion
- **langchain-ollama**: LLM integration
- **langchain-core**: LangChain core functionality
- **rich**: Terminal progress bars and formatting
- **pyyaml**: YAML configuration parsing
- **feedparser**: RSS/Atom feed parsing

### Development Dependencies

- **pytest**: Testing
- **black**: Code formatting
- **isort**: Import organization
- **flake8**: Linting
- **mypy**: Static type checking

## Notes

- The crawler respects `robots.txt` by default
- Articles are only crawled once (duplicate detection)
- LLM summarization requires local Ollama installation
- Archive directory preserves historical summaries
- Web server is for local use only (no authentication)

## License

Apache 2.0

## Author

Sergey Zhumatiy <sergzhum@gmail.com>
