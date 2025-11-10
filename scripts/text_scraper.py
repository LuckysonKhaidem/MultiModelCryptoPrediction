import csv
from datetime import datetime
from typing import List, Dict

import feedparser
from bs4 import BeautifulSoup


RSS_URLS = "rss_urls.txt"

def load_urls():
    with open(RSS_URLS) as f:
        data = f.read()
        return data.split("\n")

CRYPTO_FEEDS = load_urls()


def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(" ", strip=True)


def parse_entry(entry, source_url: str) -> Dict[str, str]:
    pub_dt = ""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        pub_dt = datetime(*entry.published_parsed[:6]).isoformat()
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        pub_dt = datetime(*entry.updated_parsed[:6]).isoformat()

    title = getattr(entry, "title", "").strip()
    summary_html = getattr(entry, "summary", "") or getattr(entry, "description", "")
    summary = clean_html(summary_html)

    link = getattr(entry, "link", "").strip()

    return {
        "published_date": pub_dt,
        "title": title,
        "summary_content": summary,
        "link": link,
        "source": source_url,
    }


def fetch_feed(url: str) -> List[Dict[str, str]]:
    print(f"Fetching: {url}")
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries:
        try:
            article = parse_entry(entry, url)
            # Skip entries with no title or link
            if not article["title"] and not article["link"]:
                continue
            articles.append(article)
        except Exception as e:
            print(f"Error parsing entry from {url}: {e}")
    return articles


def write_to_csv(filename: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No rows to write.")
        return

    fieldnames = ["published_date", "title", "summary_content", "link", "source"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {filename}")


def main():
    all_articles: List[Dict[str, str]] = []

    for feed_url in CRYPTO_FEEDS:
        articles = fetch_feed(feed_url)
        all_articles.extend(articles)

    # Optional: de-duplicate by link
    seen = set()
    unique_articles = []
    for a in all_articles:
        if a["link"] in seen:
            continue
        seen.add(a["link"])
        unique_articles.append(a)

    write_to_csv("crypto_news.csv", unique_articles)


if __name__ == "__main__":
    main()