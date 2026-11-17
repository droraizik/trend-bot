import json
from pathlib import Path
import pandas as pd


def series_to_lists(series):
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "values": [None if pd.isna(v) else round(float(v), 2) for v in series.values],
    }


def headline_for_page(headline):
    return {
        "title": headline["title"],
        "url": headline["url"],
        "source": headline["source"],
        "published": headline["published"].strftime("%b %d, %H:%M UTC"),
        "score": headline.get("score"),
    }


def render_dashboard(template_path, page_path, data):
    template = Path(template_path).read_text(encoding="utf-8")
    # keeps a headline containing "</script>" from breaking the page
    payload = json.dumps(data).replace("</", "<\\/")
    Path(page_path).parent.mkdir(parents=True, exist_ok=True)
    Path(page_path).write_text(template.replace("__BOT_DATA__", payload), encoding="utf-8")
