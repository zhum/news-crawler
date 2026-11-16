#!/usr/bin/env python3
"""fix-selectors.py — propose config.yaml fixes for broken CSS selectors.

Reads the crawl issues file written by the spider (issues_file in config,
default crawl_issues.json), fetches each failing page again, asks the
running llama.cpp server (config ai.model, e.g. localhost:8090) for new
selectors, tests them exactly the way the spider applies them, and prints
the test result plus the proposed config.yaml change when the test passes.

Usage: ./fix-selectors.py [--config config.yaml] [--llm localhost:8090]
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from urllib.parse import urlparse

import requests
import yaml
from lxml import etree, html as lhtml
from markdownify import markdownify as md
from parsel import Selector

ATTEMPTS = 3
MAX_HTML = 60_000  # chars of cleaned HTML sent to the LLM (~20k tokens)
SELECTOR_KEYS = ('css', 'link', 'itemcss', 'headercss', 'articlecss')

FIELD_HELP = {
    'css': "CSS selector matching every news item container on the listing page",
    'link': "CSS selector applied inside each item; must end with ::attr(href)",
    'itemcss': "CSS selector matching every news item on the page",
    'headercss': "CSS selector for the article title; must end with ::text",
    'articlecss': "CSS selector for the element holding the article body",
}


def clean_html(text):
    """Shrink a page so it fits the LLM context: drop scripts and styling,
    keep only class/id/href attributes, shorten long text nodes."""
    root = lhtml.fromstring(text)
    etree.strip_elements(root, 'script', 'style', 'svg', 'noscript', 'iframe',
                         'head', 'nav', 'footer', 'aside', 'form', 'select',
                         etree.Comment, with_tail=False)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for a in list(el.attrib):
            if a not in ('class', 'id', 'href'):
                del el.attrib[a]
        # selectors need structure, not text: short text keeps the prompt small
        if el.text and len(el.text) > 80:
            el.text = el.text[:80] + '…'
        if el.tail and len(el.tail) > 80:
            el.tail = el.tail[:80] + '…'
    out = lhtml.tostring(root, encoding='unicode')
    return re.sub(r'\s+', ' ', out)[:MAX_HTML]


def ask_llm(base, prompt):
    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 400,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('LLAMA_API_KEY'):
        headers['Authorization'] = f"Bearer {os.environ['LLAMA_API_KEY']}"
    req = urllib.request.Request(f"http://{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=900) as resp:
        text = json.loads(resp.read())['choices'][0]['message']['content']
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0)) if m else {}


def test(site, page, url, kind):
    """Apply selectors like news_spider does. Returns (ok, result lines)."""
    sel = Selector(text=page)
    lines = []
    if kind == 'article':
        header = (sel.css(site['headercss']).get() or '').strip()
        body = sel.css(site['articlecss']).get()
        para = next((p for p in md(body).split('\n\n') if p.strip()), '') if body else ''
        if header:
            lines.append(f"title: {header}")
        if para:
            lines.append(f"first paragraph: {para.strip()[:300]}")
        return bool(header and para), lines
    if site.get('onepage'):
        for item in sel.css(site['itemcss']):
            header = item.css(site['headercss']).get()
            if header and item.css(site['articlecss']).get():
                lines.append(f"{header.strip()[:100]}")
        return bool(lines), lines
    for node in sel.css(site['css']):
        link = node.css(site['link']).get()
        if link:
            title = ' '.join(' '.join(node.css('*::text').getall()).split())
            lines.append(f"{title[:100]} — {requests.compat.urljoin(url, link)}")
    return bool(lines), lines


def fix_site(site, kind, url, problems, llm, ua):
    print(f"\n=== {site['name']} ({kind} page) {url}")
    for p in problems:
        print(f"    issue: {p}")
    try:
        r = requests.get(url, headers={'User-Agent': ua}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"    cannot fetch page: {e}")
        return
    # shortcut: last two host labels as "site domain", wrong for co.uk-style TLDs
    def domain(u):
        return '.'.join((urlparse(u).hostname or '').split('.')[-2:])
    if domain(r.url) != domain(site['url']):
        print(f"    page redirects to {r.url} — not a selector problem "
              "(paywall/moved site?), check by hand")
        return
    fields = (['headercss', 'articlecss'] if kind == 'article'
              else ['itemcss', 'headercss', 'articlecss'] if site.get('onepage')
              else ['css', 'link'])
    page = r.text
    prompt = (
        "This page was crawled with CSS selectors (parsel/Scrapy syntax) that "
        "stopped working after a site redesign.\n"
        + ''.join(f"- {f}: currently '{site.get(f)}' — {FIELD_HELP[f]}\n" for f in fields)
        + "Problems: " + '; '.join(problems) + "\n"
        + "Reply with JSON only, with keys " + ', '.join(fields)
        + ". Prefer short stable selectors (classes or tags, not generated ids).\n"
        + f"Page URL: {r.url}\nHTML:\n{clean_html(page)}")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            proposal = ask_llm(llm, prompt)
        except Exception as e:
            print(f"    LLM request failed: {e}")
            return
        proposal = {f: proposal[f] for f in fields
                    if isinstance(proposal.get(f), str) and proposal[f].strip()}
        candidate = {**site, **proposal}
        try:
            ok, lines = test(candidate, page, r.url, kind)
        except Exception as e:  # invalid selector syntax
            ok, lines = False, [f"selector error: {e}"]
        print(f"--- attempt {attempt}: {proposal}")
        if not ok and not lines:
            print("    nothing matched")
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print(f"    … {len(lines) - 10} more")
        if ok:
            changed = {k: v for k, v in proposal.items() if v != site.get(k)}
            print("    TEST OK. Proposed config.yaml change:" if changed
                  else "    TEST OK with current selectors (transient problem?)")
            if changed:
                print(f"      - name: {site['name']}")
                for k, v in changed.items():
                    print(f"        {k}: {v}    # was: {site.get(k)}")
            return
        prompt += (f"\n\nYour previous answer {json.dumps(proposal)} failed the "
                   f"test: {lines[:3] or 'nothing matched'}. Try different selectors.")
    print("    no working selectors found")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--llm', help="llama.cpp server host:port (default: config ai.model)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress visible through tee
    with open(args.config) as f:
        config = yaml.safe_load(f)
    llm = args.llm or config.get('ai', {}).get('model', 'localhost:8090')
    if not re.fullmatch(r'[\w.-]+:\d+', llm):
        ap.error(f"--llm must be host:port of a llama.cpp server, got {llm!r}")
    ua = config.get('spider', {}).get('user_agent', 'newsfetcher')
    issues_file = config.get('issues_file', 'crawl_issues.json')
    try:
        with open(issues_file) as f:
            issues = json.load(f)
    except FileNotFoundError:
        print(f"{issues_file} not found — nothing to fix")
        return
    sites = {s['name']: s for g in config.get('groups', [{'sites': config.get('sites', [])}])
             for s in g['sites']}

    # (site, kind) -> (example url, [problems])
    todo = {}
    for i in issues:
        problem, site = i.get('problem', ''), sites.get(i.get('site'))
        if not site or not problem.startswith(tuple(f"{k} '" for k in SELECTOR_KEYS)):
            print(f"skip [{i.get('site')}]: {problem} (not a selector issue)")
            continue
        kind = ('article' if problem.startswith(('headercss', 'articlecss'))
                and not site.get('onepage') else 'listing')
        todo.setdefault((site['name'], kind), (i['url'], []))[1].append(problem)
    for (name, kind), (url, problems) in todo.items():
        fix_site(sites[name], kind, url, problems, llm, ua)


if __name__ == '__main__':
    main()
