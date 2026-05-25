from pathlib import Path
import feedparser

import os
import scrapy
import yaml
from markdownify import markdownify as md
from datetime import date

from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

CONF_FILE = "news_conf.yml"


def url2path(url):
    ret = url.replace('https://', '')
    ret = ret.replace('http://', '')
    ret = ret.replace('/', '-')
    ret = ret.replace(' ', '-')
    ret = ret.replace('\t', '-')
    return ret


class NewsSpider(scrapy.Spider):
    name = "news"

    start_urls = []

    def __init__(self, only=None, conf="config.yaml", *args, **kwargs):

        """Load configuration from config.yaml file."""
        config_path = Path(__file__).parent.parent.parent / conf

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        super().__init__(*args, **kwargs)
        self.only_sites = only

        only_sites = self.only_sites.split(',') if self.only_sites else None
        if only_sites:
            names = ','.join(only_sites)
            urls = list([x['url'] for x in only_sites])
        else:
            names = ','.join([x['name'] for x in self.config['sites']])
            urls = list([x['url'] for x in self.config['sites']])
        print(f"Crawling: {names}")
        NewsSpider.start_urls = urls

    def filename(self, url):
        base = os.environ.get('CRAWL_BASE', ".")
        u = url2path(url)
        # return f"{base}/{self.__class__.name}-{u}.txt"
        return f"{base}/{u}.txt"

    async def start(self):
        only_sites = self.only_sites.split(',') if self.only_sites else None

        if only_sites:
            names = ','.join(only_sites)
        else:
            names = ','.join([x['name'] for x in self.config['sites']])
        print(f"Start crawling: {names}")
        for site in self.config["sites"]:
            if only_sites and not site['name'] in only_sites:
                print(f"Skip {site['name']}")
                continue
            print(f"==> {site['name']} ({site['url']})")
            yield scrapy.Request(
                url=site["url"],
                callback=self.parse,
                errback=self.handle_error,
                cb_kwargs=site)
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
            for url in items:  # type: ignore
                try:
                    if url is None:
                        continue
                    filename = self.filename(url)
                    if (not Path(filename).is_file() and
                       url is not None):
                        yield response.follow(
                            url,
                            callback=self.parse_article,
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
        Path(filename).write_text(
            f"URL: {response.url}\nHeader: {header}\nText:\n{markdown}")
        info = Path(self.config.get("added", "added.txt"))
        # print("...")
        with info.open(mode="a+") as f:
            # fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
            f.write(f"{date.today()}: {filename}\n")
            # fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
        # print(f"<<< url={response.url}")

    def parse_onepage(self, response, site):
        list = response.css(site['itemcss'])
        print(f"--- one --- url={response.url} ({len(list)})")
        for item in list:
            header = item.css(site["headercss"]).get()
            text = item.css(site["articlecss"]).get()
            markdown = md(text)
            filename = self.filename(f"{site['name']}-{header[:64]}")
            Path(filename).write_text(
                f"URL: {response.url}\nHeader: {header}\nText:\n{markdown}")
            info = Path(self.config.get("added", "added.txt"))
            with info.open(mode="a+") as f:
                # fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                f.write(f"{date.today()}: {filename}\n")
                # fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
        # print(f"<<< one <<< url={response.url} ({len(list)})")
