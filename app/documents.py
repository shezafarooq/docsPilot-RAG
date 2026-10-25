"""Download selected documentation and extract article sections with source links."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

SOURCES = (
    "https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/",
    "https://docs.docker.com/storage/volumes/",
    "https://docs.docker.com/storage/bind-mounts/",
    "https://docs.docker.com/reference/compose-file/volumes/",
    "https://docs.docker.com/reference/compose-file/services/#volumes",
    "https://docs.docker.com/reference/compose-file/services/#ports",
    "https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
)
EXTRACTOR_VERSION = 2
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCKS = HEADINGS | {"p", "pre", "ul", "ol", "table", "blockquote"}


def extract_document(html: str, source_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    # Comma-separated selectors return the first match in document order, which
    # would select an enclosing <main> and accidentally include its sidebar.
    root = soup.select_one("article.md-content__inner") or soup.find("article") or soup.find("main")
    if root is None:
        raise ValueError(f"No article content found: {source_url}")
    for node in root.select("script, style, nav, aside, footer, button, svg, .headerlink, .linenos, .linenodiv, [hidden], [aria-hidden='true']"):
        node.decompose()
    # Material for MkDocs repeats examples for multiple Python versions. Keep one.
    for tab_group in root.select(".tabbed-content"):
        for variant in tab_group.select(":scope > .tabbed-block")[1:]:
            variant.decompose()
    title_node = root.find("h1") or soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else (soup.title.get_text(strip=True) if soup.title else source_url)
    page_url, fragment = urldefrag(source_url)
    sections = []
    heading, anchor, blocks = title, "", []
    active = not fragment
    selected_level = None

    def finish():
        if blocks:
            sections.append({
                "heading": heading,
                "url": page_url + (f"#{anchor}" if anchor else ""),
                "text": "\n\n".join(blocks),
            })

    for node in root.find_all(list(BLOCKS)):
        if any(parent.name in BLOCKS for parent in node.parents if parent is not root):
            continue
        if node.name in HEADINGS:
            level = int(node.name[1])
            node_anchor = node.get("id", "")
            if not node_anchor:
                link = node.find("a", href=lambda value: value and value.startswith("#"))
                node_anchor = link["href"][1:] if link else ""
            if fragment and not active:
                if node_anchor != fragment:
                    continue
                active, selected_level = True, level
            elif fragment and active and selected_level is not None and level <= selected_level:
                break
            finish()
            blocks = []
            heading = node.get_text(" ", strip=True)
            anchor = node_anchor
        elif active:
            if node.name == "pre":
                code = node.get_text("", strip=False).strip("\n")
                text = f"```\n{code}\n```"
            else:
                text = node.get_text("\n" if node.name in {"ul", "ol", "table"} else " ", strip=True)
            if text.strip():
                blocks.append(text)
    finish()
    if not sections:
        raise ValueError(f"No text found for article or requested section: {source_url}")
    return {
        "source_url": source_url,
        "title": title,
        "project": "docker" if urlsplit(source_url).hostname == "docs.docker.com" else "fastapi",
        "sections": sections,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
    }


def load_document(source_url: str, data_dir: Path, refresh: bool = False) -> dict:
    cache_dir = data_dir / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(source_url.encode()).hexdigest()[:16] + ".json"
    path = cache_dir / filename
    if path.exists() and not refresh:
        document = json.loads(path.read_text())
        if document.get("extractor_version") == EXTRACTOR_VERSION:
            print(f"Using saved document: {source_url}", flush=True)
            return document
    page_url, _ = urldefrag(source_url)
    request = Request(page_url, headers={"User-Agent": "DocsPilot/0.1 (documentation learning project)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8")
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    document = extract_document(html, source_url)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(f"Saved {len(document['sections'])} sections: {source_url}", flush=True)
    return document
