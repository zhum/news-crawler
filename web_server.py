#!/usr/bin/env python3
import json
import sys
import yaml

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8001
CONFIG = {}


def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {'output': {'summary_file': 'summary.json'}}


def get_groups(config):
    if 'groups' in config:
        return config['groups']
    return [{
        'name': 'All',
        'output': config.get('output', {'summary_file': 'summary.json'}),
    }]


def active_tag_keys(config, group_filter, all_groups):
    """Return ordered tag keys relevant to the current group filter."""
    all_tags = config.get('tags', {})
    if not all_tags:
        return []
    if group_filter:
        group = next((g for g in all_groups if g['name'] == group_filter), None)
        if group and 'tags' in group:
            return [k for k in group['tags'] if k in all_tags]
    return list(all_tags.keys())


def tag_rating(article, key):
    tags = article.get('tags', {})
    if isinstance(tags, dict):
        return tags.get(key, 0)
    return 0


class SummaryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        sort_by = query_params.get("sort", ["title"])[0]
        order = query_params.get("order", ["asc"])[0]
        group_filter = query_params.get("group", [None])[0]

        groups = get_groups(CONFIG)

        if group_filter:
            active_groups = [g for g in groups if g['name'] == group_filter]
            if not active_groups:
                active_groups = groups
        else:
            active_groups = groups

        all_data = []
        errors = []
        for group in active_groups:
            summary_file = group.get('output', {}).get('summary_file', 'summary.json')
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    item['_group'] = group['name']
                all_data.extend(data)
            except FileNotFoundError:
                errors.append(f"{summary_file} not found")
            except json.JSONDecodeError:
                errors.append(f"{summary_file}: invalid JSON")

        reverse = order == "desc"
        if sort_by == "rate":
            all_data.sort(key=lambda x: int(x.get("rate", 0)), reverse=reverse)
        elif sort_by == "url":
            all_data.sort(key=lambda x: x.get("url", ""), reverse=reverse)
        elif sort_by.startswith("tag:"):
            key = sort_by[4:]
            all_data.sort(key=lambda x: tag_rating(x, key), reverse=reverse)
        else:
            all_data.sort(key=lambda x: x.get("header", ""), reverse=reverse)

        tag_keys = active_tag_keys(CONFIG, group_filter, groups)
        html = self.generate_html(
            all_data, groups, group_filter, sort_by, order, tag_keys, errors)

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def generate_html(self, data, all_groups, active_group,
                      current_sort, current_order, tag_keys, errors=None):

        def sort_link(field, display_name):
            group_param = f"&group={active_group}" if active_group else ""
            new_order = (
                "desc" if (current_sort == field and current_order == "asc") else "asc"
            )
            arrow = (
                " ↓" if current_sort == field and current_order == "desc"
                else " ↑" if current_sort == field
                else ""
            )
            return (
                f'<a href="?sort={field}&order={new_order}{group_param}">'
                f'{display_name}{arrow}</a>'
            )

        # Group navigation
        group_nav = '<a href="?" class="group-link{0}">All</a>'.format(
            ' active' if not active_group else '')
        for g in all_groups:
            name = g['name']
            active_cls = ' active' if name == active_group else ''
            group_nav += (
                f'<a href="?group={name}" class="group-link{active_cls}">'
                f'{self.escape_html(name)}</a>'
            )

        # Tag sort links
        tag_sort_row = ''
        if tag_keys:
            tag_sort_row = '<div class="controls"><strong>Sort by tag:</strong> '
            tag_sort_row += ' '.join(
                sort_link(f"tag:{k}", k) for k in tag_keys)
            tag_sort_row += '</div>'

        show_group_badge = not active_group and len(all_groups) > 1

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white;
                      padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; margin-bottom: 20px; }}
        .group-nav {{ text-align: center; margin-bottom: 20px; padding: 10px;
                      background: #f8f9fa; border-radius: 5px; }}
        .group-link {{ margin: 0 6px; padding: 6px 14px; background: #6c757d; color: white;
                       text-decoration: none; border-radius: 4px; display: inline-block; }}
        .group-link.active {{ background: #007bff; }}
        .group-link:hover {{ opacity: 0.85; }}
        .controls {{ margin-bottom: 12px; padding: 12px 15px; background-color: #f8f9fa;
                     border-radius: 5px; text-align: center; }}
        .controls a {{ margin: 3px 6px; padding: 6px 12px; background-color: #007bff;
                        color: white; text-decoration: none; border-radius: 4px;
                        display: inline-block; font-size: 0.9em; }}
        .controls a:hover {{ background-color: #0056b3; }}
        .article {{ border: 1px solid #ddd; margin-bottom: 20px; padding: 20px;
                    border-radius: 8px; background-color: #fafafa; }}
        .article:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .rate {{ display: inline-block; background-color: #28a745; color: white;
                 padding: 4px 12px; border-radius: 20px; font-weight: bold;
                 margin-bottom: 10px; }}
        .rate.high {{ background-color: #dc3545; }}
        .rate.medium {{ background-color: #ffc107; color: #333; }}
        .group-badge {{ display: inline-block; background: #6c757d; color: white;
                        padding: 2px 10px; border-radius: 12px; font-size: 0.8em;
                        margin-left: 8px; vertical-align: middle; }}
        .header {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .url {{ color: #007bff; text-decoration: none; font-size: 0.9em; margin-bottom: 10px;
                display: block; word-break: break-all; }}
        .url:hover {{ text-decoration: underline; }}
        .summary {{ line-height: 1.6; color: #555; margin-top: 10px; }}
        .file {{ font-size: 0.8em; color: #666; margin-top: 10px; font-style: italic; }}
        .stats {{ text-align: center; margin-bottom: 20px; color: #666; font-size: 0.9em; }}
        .notice {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;
                   padding: 10px 16px; margin-bottom: 20px; color: #856404; font-size: 0.9em; }}
        .tags {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px; }}
        .tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                font-size: 0.78em; font-weight: bold; white-space: nowrap; }}
        .tag-high  {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .tag-mid   {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
        .tag-low   {{ background: #f8f9fa; color: #6c757d; border: 1px solid #dee2e6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Article Summary Viewer</h1>
        <div class="group-nav">{group_nav}</div>
"""
        if errors:
            for err in errors:
                html += f'        <div class="notice">⚠ {self.escape_html(err)}</div>\n'

        html += f"""
        <div class="stats">Total articles: {len(data)}</div>
        <div class="controls">
            <strong>Sort by:</strong>
            {sort_link("title", "Title")}
            {sort_link("url", "URL")}
            {sort_link("rate", "Rating")}
        </div>
        {tag_sort_row}
        <div class="articles">
"""
        for article in data:
            rate = int(article.get("rate", 0))
            rate_class = "high" if rate >= 9 else "medium" if rate >= 8 else ""

            group_badge = ""
            if show_group_badge and article.get('_group'):
                group_badge = (
                    f'<span class="group-badge">'
                    f'{self.escape_html(article["_group"])}</span>'
                )

            # Tag badges
            tags_html = self._render_tags(article.get('tags', {}), current_sort)

            html += f'''
            <div class="article">
                <div class="rate {rate_class}">{article.get("rate", "N/A")}</div>
                {group_badge}
                <div class="header">{self.escape_html(article.get("header", "No title"))}</div>
                <a href="{article.get("url", "#")}" target="_blank" class="url">{self.escape_html(article.get("url", "No URL"))}</a>
                <div class="summary">{self.escape_html(article.get("summary", "No summary"))}</div>
                <div class="tags">{tags_html}</div>
                <div class="file">Source: {self.escape_html(article.get("file", "Unknown"))}</div>
            </div>
'''  # noqa E501

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

    def _render_tags(self, tags, current_sort):
        if not tags:
            return ""
        if isinstance(tags, str):
            return self.escape_html(tags)

        active_tag = current_sort[4:] if current_sort.startswith("tag:") else None

        # Sort: active sort tag first, then by rating descending
        items = sorted(tags.items(),
                       key=lambda kv: (kv[0] != active_tag, -kv[1]))

        parts = []
        for name, rating in items:
            if rating >= 7:
                cls = "tag tag-high"
            elif rating >= 4:
                cls = "tag tag-mid"
            else:
                cls = "tag tag-low"
            bold = ' style="outline:2px solid #333;"' if name == active_tag else ''
            parts.append(
                f'<span class="{cls}"{bold}>'
                f'{self.escape_html(name)}: {rating}</span>'
            )
        return ''.join(parts)

    def escape_html(self, text):
        if not text:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


def run_server(port=PORT, config_path='config.yaml'):
    global CONFIG
    CONFIG = load_config(config_path)
    server_address = ("", port)
    httpd = HTTPServer(server_address, SummaryHandler)
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    config_path = 'config.yaml'
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        config_path = sys.argv[2] if len(sys.argv) > 2 else 'config.yaml'
    elif len(sys.argv) > 1:
        config_path = sys.argv[1]
    run_server(config_path=config_path)
