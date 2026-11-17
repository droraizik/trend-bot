import time
import xml.etree.ElementTree as ET
import requests


def sec_get(url, user_agent):
    # the SEC asks for at most 10 requests a second and a contact in the User-Agent
    time.sleep(0.2)
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()
    return response


def list_13f_filings(cik, since, user_agent):
    submissions = sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json", user_agent).json()
    older_pages = submissions["filings"].get("files", [])
    batches = [submissions["filings"]["recent"]] + [sec_get(f"https://data.sec.gov/submissions/{p['name']}", user_agent).json() for p in older_pages]
    rows = [row for batch in batches for row in rows_of(batch)]
    return [{"accession": r["accessionNumber"], "filed": r["filingDate"], "period": r.get("reportDate", "")}
            for r in rows if r["form"] == "13F-HR" and r["filingDate"] >= since]


def rows_of(batch):
    return [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]


def fetch_13f_holdings(cik, accession, user_agent):
    folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    items = sec_get(folder + "/index.json", user_agent).json()["directory"]["item"]
    table_name = next(i["name"] for i in items if i["name"].lower().endswith(".xml") and i["name"].lower() != "primary_doc.xml")
    return holdings_from_table(ET.fromstring(sec_get(f"{folder}/{table_name}", user_agent).content))


def holdings_from_table(root):
    holdings = {}
    for row in root.iter():
        if local_name(row.tag) != "infoTable" or child_text(row, "putCall"):
            continue
        entry = holdings.setdefault(child_text(row, "cusip").upper(), {"name": child_text(row, "nameOfIssuer"), "value": 0.0})
        entry["value"] += float(child_text(row, "value") or 0)
    return holdings


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def child_text(element, name):
    found = next((c for c in element.iter() if local_name(c.tag) == name), None)
    return (found.text or "").strip() if found is not None else ""


def request_figi_tickers(cusips):
    jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in cusips]
    try:
        response = requests.post("https://api.openfigi.com/v3/mapping", json=jobs, timeout=30)
        response.raise_for_status()
        return dict(zip(cusips, [yahoo_symbol(result) for result in response.json()]))
    except Exception as error:
        print(f"Ticker lookup failed for {len(cusips)} CUSIPs, will retry next run: {error}")
        return {}


def yahoo_symbol(result):
    matches = result.get("data") or []
    ticker = matches[0].get("ticker") if matches else None
    return ticker.replace("/", "-").replace(".", "-") if ticker else None


def holdings_to_weights(holdings, cusip_tickers):
    values = {}
    for cusip, entry in holdings.items():
        ticker = cusip_tickers.get(cusip)
        if ticker:
            values[ticker] = values.get(ticker, 0.0) + entry["value"]
    total = sum(values.values())
    return {t: v / total for t, v in values.items()} if total else {}


def holdings_names(holdings, cusip_tickers):
    return {cusip_tickers[c]: e["name"].title() for c, e in holdings.items() if cusip_tickers.get(c)}
