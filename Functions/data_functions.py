from pathlib import Path
import pandas as pd
import yfinance as yf


def download_closing_prices(tickers, start_date, warmup_days):
    fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)
    data = yf.download(tickers, start=fetch_start, auto_adjust=True, progress=False)
    # Yahoo sometimes leaves a stock's latest price blank; the last known price stands in for it
    prices = data["Close"].dropna(how="all").ffill()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices


def fetch_recent_headlines(ticker, max_items, max_age_days):
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as error:
        print(f"News for {ticker} unavailable: {error}")
        return []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
    headlines = [parse_news_item(item) for item in raw_items]
    recent = [h for h in headlines if h["title"] and h["published"] >= cutoff]
    return sorted(recent, key=lambda h: h["published"], reverse=True)[:max_items]


# Yahoo has changed its news format before, so both the old and new shapes are read
def parse_news_item(item):
    content = item.get("content") or item
    url = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
    source = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
    published = content.get("pubDate") or item.get("providerPublishTime")
    return {"title": content.get("title", ""), "url": url, "source": source, "published": to_utc_timestamp(published)}


def to_utc_timestamp(value):
    if value is None:
        return pd.Timestamp(0, tz="UTC")
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s", tz="UTC")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def read_decision_log(log_path):
    if not Path(log_path).exists():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str), "weight": pd.Series(dtype=float)})
    return pd.read_csv(log_path, parse_dates=["date"])


def save_decision_log(log, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log.sort_values(["date", "ticker"]).to_csv(log_path, index=False, date_format="%Y-%m-%d")
