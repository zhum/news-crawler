from pathlib import Path
import feedparser

import os
import re
import json
import scrapy
import yaml
from markdownify import markdownify as md
from datetime import date

from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

CONF_FILE = "news_conf.yml"

_URL2PATH_SCHEME_RE = re.compile(r'^https?://')
_URL2PATH_SEP_RE = re.compile(r'[/ \t]')


def url2path(url):
    ret = _URL2PATH_SCHEME_RE.sub('', url)
    return _URL2PATH_SEP_RE.sub('-', ret)


class NewsSpider(scrapy.Spider):
    name = "news"

    start_urls = []

    def __init__(self, only=None, group=None, conf="config.yaml", *args, **kwargs):

        """Load configuration from config.yaml file."""
        config_path = Path(__file__).parent.parent.parent / conf

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        super().__init__(*args, **kwargs)
        self.only_sites = only
        self.only_groups = group
        self.issues = {}  # (site, problem) -> issue record
        self.crawled_sites = set()

        only_groups = self.only_groups.split(',') if self.only_groups else None
        only_sites = self.only_sites.split(',') if self.only_sites else None
        all_sites = [
            s for g in self._get_groups()
            if not only_groups or g['name'] in only_groups
            for s in g['sites']
        ]
        if only_sites:
            all_sites = [s for s in all_sites if s['name'] in only_sites]
        names = ', '.join(s['name'] for s in all_sites)
        print(f"Crawling: {names}")
        NewsSpider.start_urls = [s['url'] for s in all_sites]

    def _get_groups(self):
        if 'groups' in self.config:
            return self.config['groups']
        return [{
            'name': 'default',
            'sites': self.config.get('sites', []),
            'output': self.config.get('output', {}),
        }]

    def filename(self, url):
        base = os.environ.get('CRAWL_BASE', ".")
        u = url2path(url)
        # return f"{base}/{self.__class__.name}-{u}.txt"
        return f"{base}/{u}.txt"

    async def start(self):
        only_groups = self.only_groups.split(',') if self.only_groups else None
        only_sites = self.only_sites.split(',') if self.only_sites else None
        groups = self._get_groups()
        names = []
        for group in groups:
            if only_groups and group.get('name') not in only_groups:
                continue
            added_file = group.get('output', {}).get(
                'added_file', self.config.get('added', 'added.txt'))
            group_spider = group.get('spider', {})
            group_ua = group_spider.get('user_agent')
            group_headers = {**({'User-Agent': group_ua} if group_ua else {}),
                             **group_spider.get('headers', {})}
            for site in group['sites']:
                if only_sites and site['name'] not in only_sites:
                    print(f"Skip {site['name']}")
                    continue
                names.append(site['name'])
                self.crawled_sites.add(site['name'])
                print(f"==> {site['name']} ({site['url']})")
                headers = {**group_headers, **site.get('headers', {})}
                yield scrapy.Request(
                    url=site["url"],
                    callback=self.parse,
                    errback=self.handle_error,
                    headers=headers,
                    cb_kwargs={**site, '_added_file': added_file,
                                '_headers': headers,
                                '_group': group.get('name')})
        print(f"Start crawling: {', '.join(names)}")
        print("End crawling")

    def handle_error(self, failure):
        # log all failures
        self.logger.error(repr(failure))

        # in case you want to do something special for some errors,
        # you may need the failure's type:

        if failure.check(HttpError):
            # these exceptions come from HttpError spider middleware
            # you can get the non-200 response
            response = failure.value.response
            self.logger.error("HttpError on %s", response.url)

        elif failure.check(DNSLookupError):
            # this is the original request
            request = failure.request
            self.logger.error("DNSLookupError on %s", request.url)

        elif failure.check(TimeoutError, TCPTimedOutError):
            request = failure.request
            self.logger.error("TimeoutError on %s", request.url)

        site = failure.request.cb_kwargs
        if 'name' in site:
            self._issue(site, f"fetch failed: {failure.value!r}"[:200],
                        failure.request.url)

    def _issue(self, site, problem, url):
        """Record a crawl problem (e.g. selector no longer matching after a
        site redesign) so it shows in the log and on the web server."""
        self.logger.warning("ISSUE [%s] %s (%s)", site['name'], problem, url)
        rec = self.issues.setdefault((site['name'], problem), {
            'date': str(date.today()), 'group': site.get('_group'),
            'site': site['name'], 'problem': problem, 'url': url, 'count': 0})
        rec['count'] += 1

    def closed(self, reason):
        # Replace previous issues of the sites crawled now; keep the others
        path = Path(self.config.get('issues_file', 'crawl_issues.json'))
        try:
            old = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            old = []
        kept = [i for i in old if i.get('site') not in self.crawled_sites]
        path.write_text(json.dumps(kept + list(self.issues.values()), indent=2))
        if self.issues:
            self.logger.warning("%d crawl issue(s) written to %s",
                                len(self.issues), path)

    def _existing_files(self):
        base = os.environ.get('CRAWL_BASE', ".")
        try:
            return {entry.name for entry in os.scandir(base) if entry.is_file()}
        except FileNotFoundError:
            return set()

    def _save_article(self, filename, url, header, markdown):
        Path(filename).write_text(
            f"URL: {url}\nHeader: {header}\nText:\n{markdown}")

    def _append_added(self, added_file, lines):
        if not lines:
            return
        with Path(added_file).open(mode="a+") as f:
            f.writelines(lines)

    def parse(self, response, **kwargs):
        try:
            site = kwargs
            print(f"!--> OPENED {site['url']}")
            if site.get('onepage'):
                # print(f"One page {site['name']}")
                self.parse_onepage(response, site)
                return
            else:
                # print(f"url={site['url']} css={site['css']}")
                if site['css'] == 'rss':
                    entries = feedparser.parse(response.body).entries
                    if not entries:
                        self._issue(site, "RSS feed has no entries", response.url)
                    items = [e.get(site['link']) for e in entries]
                else:
                    nodes = response.css(site['css'])
                    if not nodes:
                        self._issue(site, f"css '{site['css']}' matched nothing",
                                    response.url)
                    items = [e.css(site['link']).get() for e in nodes]
                if items and not any(items):
                    self._issue(site, f"link '{site['link']}' matched nothing",
                                response.url)

            print(f"{len(items)}")  # type: ignore
            existing = self._existing_files()
            for url in items:  # type: ignore
                try:
                    if url is None:
                        continue
                    filename = self.filename(url)
                    if Path(filename).name not in existing:
                        yield response.follow(
                            url,
                            callback=self.parse_article,
                            headers=site.get('_headers', {}),
                            cb_kwargs=site)
                    else:
                        print(f"==> Skipped \"{url}\"")
                except Exception as e:
                    print(f"exception! ({url}): {e}")
            # print(f"---> CLOSED {site['url']} ")
        except Exception as e:
            print(f"Parser failed: {e}")
            self._issue(site, f"parser failed: {e}", response.url)

    def parse_article(self, response, **kwargs):
        site = kwargs
        print(f"--- url={response.url}")
        header = response.css(site["headercss"]).get()
        if header:
            header = header.strip()
        else:
            self._issue(site, f"headercss '{site['headercss']}' matched nothing",
                        response.url)
        text = response.css(site["articlecss"]).get()
        if not text:
            # don't save: file absence lets the next crawl retry it
            self._issue(site, f"articlecss '{site['articlecss']}' matched nothing",
                        response.url)
            return
        markdown = md(text)
        filename = self.filename(response.url)
        self._save_article(filename, response.url, header, markdown)
        added_file = site.get('_added_file', self.config.get('added', 'added.txt'))
        self._append_added(added_file, [f"{date.today()}: {filename}\n"])

    def parse_onepage(self, response, site):
        list = response.css(site['itemcss'])
        print(f"--- one --- url={response.url} ({len(list)})")
        if not list:
            self._issue(site, f"itemcss '{site['itemcss']}' matched nothing",
                        response.url)
        added_file = site.get('_added_file', self.config.get('added', 'added.txt'))
        today = date.today()
        added_lines = []
        for item in list:
            header = item.css(site["headercss"]).get()
            text = item.css(site["articlecss"]).get()
            if not header or not text:
                which = 'headercss' if not header else 'articlecss'
                self._issue(site, f"{which} '{site[which]}' matched nothing",
                            response.url)
                continue
            markdown = md(text)
            filename = self.filename(f"{site['name']}-{header[:64]}")
            self._save_article(filename, response.url, header, markdown)
            added_lines.append(f"{today}: {filename}\n")
        self._append_added(added_file, added_lines)
