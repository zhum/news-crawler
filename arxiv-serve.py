#!/usr/bin/env python3

"""
ArXiv Summaries Web Server

Simple HTTP server for viewing ArXiv article summaries with rankings.
Serves data from both arxiv_summaries_ranked.json and arxiv_summaries.json
in separate tabs with sorting capabilities.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class ArxivServerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ArXiv summaries viewer."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.serve_html()
        elif parsed_path.path == '/data':
            # Serve JSON data based on query parameter
            query = parse_qs(parsed_path.query)
            source = query.get('source', ['ranked'])[0]
            self.serve_json(source)
        else:
            self.send_error(404, "File not found")

    def serve_html(self):
        """Serve the main HTML page."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv Summaries Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .tabs {
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #ddd;
        }

        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            font-size: 1.1em;
            transition: all 0.3s;
            border: none;
            background: transparent;
            color: #666;
        }

        .tab:hover {
            background: #e0e0e0;
        }

        .tab.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }

        .controls {
            padding: 20px 30px;
            background: #fafafa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }

        .sort-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .sort-controls label {
            font-weight: 600;
            color: #555;
        }

        .sort-controls select {
            padding: 8px 15px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            cursor: pointer;
            background: white;
            transition: all 0.3s;
        }

        .sort-controls select:hover {
            border-color: #667eea;
        }

        .stats {
            color: #666;
            font-weight: 500;
        }

        .content {
            padding: 30px;
            min-height: 400px;
        }

        .article {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }

        .article:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-color: #667eea;
        }

        .article-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
            gap: 20px;
        }

        .article-title {
            flex: 1;
        }

        .article-title h3 {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 8px;
            line-height: 1.4;
        }

        .article-title a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }

        .article-title a:hover {
            text-decoration: underline;
        }

        .article-meta {
            color: #777;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .article-categories {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }

        .category-tag {
            background: #f0f0f0;
            color: #555;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }

        .article-summary {
            color: #555;
            line-height: 1.6;
            margin: 15px 0;
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }

        .rankings {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }

        .ranking-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.9em;
        }

        .ranking-label {
            color: #666;
            font-weight: 500;
        }

        .ranking-score {
            font-weight: 700;
            font-size: 1.1em;
            padding: 4px 10px;
            border-radius: 4px;
        }

        .score-high {
            background: #bff7b5;
            color: white;
        }

        .score-high-current {
            background: #4caf50;
            color: white;
        }

        .score-medium {
            background: #f7dea8;
            color: white;
        }

        .score-medium-current {
            background: #ff9800;
            color: white;
        }

        .score-low {
            background: #f7b5b5;
            color: white;
        }

        .score-low-current {
            background: #f44336;
            color: white;
        }

        .loading {
            text-align: center;
            padding: 60px;
            color: #999;
            font-size: 1.2em;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #c62828;
        }

        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: stretch;
            }

            .rankings {
                grid-template-columns: 1fr;
            }

            header h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ArXiv Summaries Viewer</h1>
            <p>Browse and analyze ArXiv article summaries with AI-powered rankings</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('ranked')">Ranked Summaries</button>
            <button class="tab" onclick="switchTab('plain')">Plain Summaries</button>
        </div>

        <div class="controls">
            <div class="sort-controls">
                <label for="sortBy">Sort by:</label>
                <select id="sortBy" onchange="sortArticles()">
                    <option value="title">Title (A-Z)</option>
                    <option value="title-desc">Title (Z-A)</option>
                    <option value="date">Date (Newest)</option>
                    <option value="date-asc">Date (Oldest)</option>
                    <option value="avg-rank">Avg Rank (High-Low)</option>
                    <option value="avg-rank-asc">Avg Rank (Low-High)</option>
                    <!-- Individual ranking options will be added dynamically -->
                </select>
            </div>
            <div class="stats" id="stats">Loading...</div>
        </div>

        <div class="content" id="content">
            <div class="loading">Loading articles...</div>
        </div>
    </div>

    <script>
        let currentTab = 'ranked';
        let articles = [];
        let rankingAreas = [];

        async function loadData(source) {
            try {
                const response = await fetch(`/data?source=${source}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                articles = await response.json();

                // Extract ranking areas from the first article with rankings
                rankingAreas = [];
                for (const article of articles) {
                    if (article.rankings) {
                        rankingAreas = Object.keys(article.rankings);
                        break;
                    }
                }

                updateSortOptions();
                sortArticles();
            } catch (error) {
                document.getElementById('content').innerHTML =
                    `<div class="error"><strong>Error loading data:</strong> ${error.message}</div>`;
            }
        }

        function updateSortOptions() {
            const sortBy = document.getElementById('sortBy');
            const currentValue = sortBy.value;

            // Remove all individual ranking options (they start with 'rank-')
            const options = Array.from(sortBy.options);
            options.forEach(opt => {
                if (opt.value.startsWith('rank-')) {
                    sortBy.removeChild(opt);
                }
            });

            // Add individual ranking options if in ranked tab
            if (currentTab === 'ranked' && rankingAreas.length > 0) {
                rankingAreas.forEach(area => {
                    const optHigh = document.createElement('option');
                    optHigh.value = `rank-${area}`;
                    optHigh.textContent = `${area} (High-Low)`;
                    optHigh.style.display = 'block';
                    sortBy.appendChild(optHigh);

                    const optLow = document.createElement('option');
                    optLow.value = `rank-${area}-asc`;
                    optLow.textContent = `${area} (Low-High)`;
                    optLow.style.display = 'block';
                    sortBy.appendChild(optLow);
                });
            }

            // Show/hide avg rank options
            const avgRankOptions = Array.from(sortBy.options).filter(opt =>
                opt.value === 'avg-rank' || opt.value === 'avg-rank-asc'
            );
            avgRankOptions.forEach(opt => {
                opt.style.display = currentTab === 'ranked' ? 'block' : 'none';
            });

            // Restore previous value if still valid, otherwise reset
            if (currentTab === 'plain' && (currentValue.startsWith('avg-rank') || currentValue.startsWith('rank-'))) {
                sortBy.value = 'title';
            } else if (Array.from(sortBy.options).some(opt => opt.value === currentValue)) {
                sortBy.value = currentValue;
            }
        }

        function switchTab(tab) {
            currentTab = tab;

            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');

            loadData(tab);
        }

        function getScoreClass(value,key) {
            cur = document.getElementById('sortBy').value;
            // console.log("value:", value, "cur:", cur, "key:", key);
            cur = cur.replace('rank-', '');  // Remove 'rank-' if present
            if (key == cur) {
                if (value >= 8) return 'score-high-current';
                if (value >= 6) return 'score-medium-current';
                return 'score-low-current';
            }
            if (value >= 8) return 'score-high';
            if (value >= 6) return 'score-medium';
            return 'score-low';
        }

        function calculateAvgRank(article) {
            if (!article.rankings) return 0;
            const scores = Object.values(article.rankings);
            return scores.reduce((a, b) => a + b, 0) / scores.length;
        }

        function getRankScore(article, rankingArea) {
            if (!article.rankings || !article.rankings[rankingArea]) {
                return 0;
            }
            return article.rankings[rankingArea];
        }

        function sortArticles() {
            const sortByValue = document.getElementById('sortBy').value;

            const sorted = [...articles].sort((a, b) => {
                // Handle individual ranking sorts
                if (sortByValue.startsWith('rank-')) {
                    const isAscending = sortByValue.endsWith('-asc');
                    const rankingArea = isAscending
                        ? sortByValue.substring(5, sortByValue.length - 4)  // Remove 'rank-' and '-asc'
                        : sortByValue.substring(5);  // Remove 'rank-'

                    const scoreA = getRankScore(a, rankingArea);
                    const scoreB = getRankScore(b, rankingArea);

                    return isAscending ? scoreA - scoreB : scoreB - scoreA;
                }

                // Handle standard sorts
                switch(sortByValue) {
                    case 'title':
                        return a.title.localeCompare(b.title);
                    case 'title-desc':
                        return b.title.localeCompare(a.title);
                    case 'date':
                        return new Date(b.date || 0) - new Date(a.date || 0);
                    case 'date-asc':
                        return new Date(a.date || 0) - new Date(b.date || 0);
                    case 'avg-rank':
                        return calculateAvgRank(b) - calculateAvgRank(a);
                    case 'avg-rank-asc':
                        return calculateAvgRank(a) - calculateAvgRank(b);
                    default:
                        return 0;
                }
            });

            renderArticles(sorted);
        }

        function renderArticles(articlesToRender) {
            const content = document.getElementById('content');
            const stats = document.getElementById('stats');

            stats.textContent = `Showing ${articlesToRender.length} articles`;

            if (articlesToRender.length === 0) {
                content.innerHTML = '<div class="loading">No articles found</div>';
                return;
            }

            content.innerHTML = articlesToRender.map(article => {
                const avgRank = calculateAvgRank(article);

                return `
                    <div class="article">
                        <div class="article-header">
                            <div class="article-title">
                                <h3><a href="${article.url}" target="_blank">${article.title}</a></h3>
                                <div class="article-meta">
                                    ${article.arxiv_id ? `<strong>arXiv:</strong> ${article.arxiv_id} | ` : ''}
                                    ${article.date ? `<strong>Date:</strong> ${article.date}` : ''}
                                </div>
                                ${article.authors ? `<div class="article-meta"><strong>Authors:</strong> ${article.authors}</div>` : ''}
                                ${article.categories ? `
                                    <div class="article-categories">
                                        ${article.categories.split(' ').map(cat =>
                                            `<span class="category-tag">${cat}</span>`
                                        ).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        </div>

                        ${article.summary ? `
                            <div class="article-summary">
                                <strong>Summary:</strong> ${article.summary}
                            </div>
                        ` : ''}

                        ${article.rankings ? `
                            <div class="rankings">
                                ${Object.entries(article.rankings).map(([key, value]) => `
                                    <div class="ranking-item">
                                        <span class="ranking-label">${key}</span>
                                        <span class="ranking-score ${getScoreClass(value,key)}">${value}/10</span>
                                    </div>
                                `).join('')}
                                ${avgRank > 0 ? `
                                    <div class="ranking-item" style="grid-column: 1 / -1; border: 2px solid #667eea;">
                                        <span class="ranking-label" style="font-weight: 700;">Average Rank</span>
                                        <span class="ranking-score ${getScoreClass(avgRank)}">${avgRank.toFixed(2)}/10</span>
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('');
        }

        // Load initial data
        loadData('ranked');
    </script>
</body>
</html>"""  # type: ignore

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_json(self, source):
        """Serve JSON data based on source type."""
        try:
            if source == 'ranked':
                filepath = Path('arxiv_summaries_ranked.json')
            else:
                filepath = Path('arxiv_summaries.json')

            if not filepath.exists():
                self.send_error(404, f"File not found: {filepath}")
                return

            data = json.loads(filepath.read_text())

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        except Exception as e:
            self.send_error(500, f"Error loading data: {str(e)}")

    def log_message(self, format, *args):
        """Custom log message format."""
        print(f"[ArxivServer] {format % args}")


def run_server(port=8002):
    """Start the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ArxivServerHandler)
    print(f"ArXiv Summaries Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == '__main__':
    run_server()
