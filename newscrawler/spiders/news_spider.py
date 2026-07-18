from pathlib import Path
import feedparser

import os
import re
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
            group_ua = group.get('spider', {}).get('user_agent')
            for site in group['sites']:
                if only_sites and site['name'] not in only_sites:
                    print(f"Skip {site['name']}")
                    continue
                names.append(site['name'])
                print(f"==> {site['name']} ({site['url']})")
                headers = {'User-Agent': group_ua} if group_ua else {}
                yield scrapy.Request(
                    url=site["url"],
                    callback=self.parse,
                    errback=self.handle_error,
                    headers=headers,
                    cb_kwargs={**site, '_added_file': added_file,
                                '_user_agent': group_ua})
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
            else:
                # print(f"url={site['url']} css={site['css']}")
                if site['css'] == 'rss':
                    items = list(map(
                        lambda e: e.get(site['link']),
                        feedparser.parse(response.body).entries
                    ))
                    # print(f">> {items}")
                else:
                    # print(f">> {response.css(site['css'])}")
                    items = list(map(
                        lambda e: e.css(site['link']).get(),
                        response.css(site['css'])
                    ))
                    # print(f">>>> {items}")

            print(f"{len(items)}")  # type: ignore
            existing = self._existing_files()
            for url in items:  # type: ignore
                try:
                    if url is None:
                        continue
                    filename = self.filename(url)
                    if Path(filename).name not in existing:
                        ua = site.get('_user_agent')
                        yield response.follow(
                            url,
                            callback=self.parse_article,
                            headers={'User-Agent': ua} if ua else {},
                            cb_kwargs=site)
                    else:
                        print(f"==> Skipped \"{url}\"")
                except Exception as e:
                    print(f"exception! ({url}): {e}")
            # print(f"---> CLOSED {site['url']} ")
        except Exception as e:
            print(f"Parser failed: {e}")
            pass

    def parse_article(self, response, **kwargs):
        site = kwargs
        print(f"--- url={response.url}")
        header = response.css(site["headercss"]).get()
        if header:
            header = header.strip()
        text = response.css(site["articlecss"]).get()
        markdown = md(text)
        filename = self.filename(response.url)
        self._save_article(filename, response.url, header, markdown)
        added_file = site.get('_added_file', self.config.get('added', 'added.txt'))
        self._append_added(added_file, [f"{date.today()}: {filename}\n"])

    def parse_onepage(self, response, site):
        list = response.css(site['itemcss'])
        print(f"--- one --- url={response.url} ({len(list)})")
        added_file = site.get('_added_file', self.config.get('added', 'added.txt'))
        today = date.today()
        added_lines = []
        for item in list:
            header = item.css(site["headercss"]).get()
            text = item.css(site["articlecss"]).get()
            markdown = md(text)
            filename = self.filename(f"{site['name']}-{header[:64]}")
            self._save_article(filename, response.url, header, markdown)
            added_lines.append(f"{today}: {filename}\n")
        self._append_added(added_file, added_lines)
