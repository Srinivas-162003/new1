import datetime as dt
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests


def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    results: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        link = ""
        for link_elem in entry.findall("atom:link", ns):
            if link_elem.attrib.get("rel") == "alternate":
                link = link_elem.attrib.get("href", "")
                break
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = ""
        if published:
            try:
                year = str(dt.datetime.fromisoformat(published.replace("Z", "+00:00")).year)
            except ValueError:
                year = ""

        results.append({"title": title, "summary": summary, "url": link, "year": year})

    return results


def format_arxiv_results(results: List[Dict[str, str]]) -> str:
    lines = []
    for item in results:
        year = f" ({item['year']})" if item.get("year") else ""
        lines.append(f"- {item['title']}{year}: {item['url']}")
    return "\n".join(lines)
