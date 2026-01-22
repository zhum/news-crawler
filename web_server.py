#!/usr/bin/env python3
import json

# import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT=8001


class SummaryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Get sorting parameter
        sort_by = query_params.get("sort", ["title"])[0]
        order = query_params.get("order", ["asc"])[0]

        try:
            with open("summary.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            self.send_error(404, "summary.json not found")
            return
        except json.JSONDecodeError:
            self.send_error(500, "Invalid JSON format")
            return

        # Sort data
        reverse = order == "desc"
        if sort_by == "rate":
            data.sort(key=lambda x: int(x.get("rate", 0)), reverse=reverse)
        elif sort_by == "url":
            data.sort(key=lambda x: x.get("url", ""), reverse=reverse)
        else:  # title (header)
            data.sort(key=lambda x: x.get("header", ""), reverse=reverse)

        # Generate HTML
        html = self.generate_html(data, sort_by, order)

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def generate_html(self, data, current_sort, current_order):
        # Helper function to generate sort links
        def sort_link(field, display_name):
            new_order = (
                "desc" if (current_sort == field and
                           current_order == "asc") else "asc"
            )
            arrow = (
                " ↓"
                if current_sort == field and current_order == "desc"
                else " ↑"
                if current_sort == field
                else ""
            )
            return (
                f'<a href="?sort={field}&order={new_order}">'
                f'{display_name}{arrow}</a>'
            )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5; /* Light background */
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white; /* Light container background */
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333; /* Dark heading text */
            text-align: center;
            margin-bottom: 30px;
        }}
        .controls {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa; /* Light controls background */
            border-radius: 5px;
            text-align: center;
        }}
        .controls a {{
            margin: 0 15px;
            padding: 8px 16px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            display: inline-block;
        }}
        .controls a:hover {{
            background-color: #0056b3;
        }}
        .article {{
            border: 1px solid #ddd;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            background-color: #fafafa; /* Light article background */
        }}
        .article:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .rate {{
            display: inline-block;
            background-color: #28a745;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .rate.high {{
            background-color: #dc3545;
        }}
        .rate.medium {{
            background-color: #ffc107;
            color: #333;
        }}
        .header {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50; /* Dark heading text */
            margin-bottom: 10px;
        }}
        .url {{
            color: #007bff;
            text-decoration: none;
            font-size: 0.9em;
            margin-bottom: 10px;
            display: block;
            word-break: break-all;
        }}
        .url:hover {{
            text-decoration: underline;
        }}
        .summary {{
            line-height: 1.6;
            color: #555;
            margin-top: 10px;
        }}
        .file {{
            font-size: 0.8em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Article Summary Viewer</h1>

        <div class="stats">
            Total articles: {len(data)}
        </div>

        <div class="controls">
            <strong>Sort by:</strong>
            {sort_link("title", "Title")}
            {sort_link("url", "URL")}
            {sort_link("rate", "Rating")}
        </div>

        <div class="articles">
"""

        for article in data:
            rate = int(article.get("rate", 0))
            rate_class = "high" if rate >= 9 else "medium" if rate >= 8 else ""

            html += f'''
            <div class="article">
                <div class="rate {rate_class}">{article.get("rate", "N/A")}</div>
                <div class="header">{self.escape_html(article.get("header", "No title"))}</div>
                <a href="{article.get("url", "#")}" target="_blank" class="url">{self.escape_html(article.get("url", "No URL"))}</a>
                <div class="summary">{self.escape_html(article.get("summary", "No summary"))}</div>
                <div class="file">Source: {self.escape_html(article.get("file", "Unknown"))}</div>
                <div class="tags">Tags: {self.escape_html(article.get("tags", "No tags"))}</div>
            </div>
'''  # noqa E501

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

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


def run_server(port=PORT):
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
    run_server()
