import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

USER_AGENT_FILE = (
    Path(__file__).parent / '../../../../../static/fake_useragent.json'
).resolve()
BROWSER = 'chrome'


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length with ellipsis."""
    return text if len(text) <= max_length else text[:max_length] + '...'


def resolve_text_from_result(
    result: Dict[str, Any],
    crawl: bool = False,
    max_crawl_tokens: int = 1000,
    *,
    url_key: str = 'url',
    fallback_keys: Optional[List[str]] = None,
) -> str:
    """
    Extract clean text from a result dictionary by joining fallback fields
    or optionally crawling the source URL.

    Args:
        result: Dictionary from a search/crawl/extract API.
        crawl: If True, attempt to crawl the page content.
        max_crawl_tokens: Max tokens to return from crawled content.
        url_key: Key to locate the URL in the result.
        fallback_keys: Keys to consider for fallback extraction.

    Returns:
        Combined cleaned text from fallback fields or crawled content.
    """
    fallback_keys = fallback_keys or [
        'title',
        'snippet',
        'rich_snippet',
        'rich_snippet_table',
        'content',
        'raw_content',
        'description',
        'snippets',
    ]

    source_url = result.get(url_key) or result.get('link')

    if crawl and source_url:
        try:
            return extract_clean_text_from_url(source_url)[:max_crawl_tokens]
        except Exception:
            pass  # Fallback to field extraction

    def clean_value(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return '\n'.join(
                str(item).strip()
                for item in value
                if isinstance(item, (str, int, float)) and str(item).strip()
            )
        return ''

    text_parts = []
    for key in fallback_keys:
        cleaned = clean_value(result.get(key))
        if cleaned:
            text_parts.append(cleaned)

    return '\n'.join(text_parts).strip()


def extract_clean_text_from_url(url: str) -> str:
    from bs4 import BeautifulSoup, Tag

    user_agent_data = json.load(open(USER_AGENT_FILE, 'r'))
    user_agent = random.choice(user_agent_data[BROWSER])

    headers = {'User-Agent': str(user_agent), 'Content-Type': 'application/json'}
    response = requests.get(url, timeout=3, headers=headers)

    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted tags
    for tag in soup(
        [
            'script',
            'style',
            'nav',
            'footer',
            'header',
            'form',
            'aside',
            'noscript',
            'svg',
            'img',
            'iframe',
        ]
    ):
        tag.decompose()

    # Define tags to keep for main content
    valid_tags = {'p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5'}

    def is_visible(tag: Tag) -> bool:
        return tag.name in valid_tags and tag.get_text(strip=True)

    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'[^ -~]+', ' ', text)
        return text.strip()

    # Extract visible, cleaned text lines
    lines = []
    for tag in soup.find_all(is_visible):
        raw = tag.get_text(separator=' ', strip=True)
        cleaned = clean_text(raw)
        if cleaned and len(cleaned) > 30:  # Filter out tiny/empty blocks
            lines.append(cleaned)

    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)

    return '\n\n'.join(deduped)
