import os
import time
from datetime import datetime, timezone
import pandas as pd
from Functions.data_functions import (download_closing_prices, fetch_recent_headlines, read_decision_log, save_decision_log,
                                      read_json, write_json)
from Functions.signal_functions import is_above_moving_average, equal_weights_for_selected, apply_news_veto, stock_status
from Functions.news_functions import score_headlines, attach_scores
from Functions.sec_functions import (list_13f_filings, fetch_13f_holdings, request_figi_tickers, holdings_to_weights,
                                     holdings_names)
from Functions.simulation_functions import simulate_portfolio, simulate_buy_and_hold
from Functions.math_functions import performance_summary, percent_vs
from Functions.page_functions import series_to_lists, headline_for_page, render_dashboard

# Market
TECH_STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
    "META": "Meta", "AVGO": "Broadcom", "AMD": "AMD", "ORCL": "Oracle", "ADBE": "Adobe",
}
BENCHMARK = "QQQ"
SP500 = "SPY"
START_DATE = "2018-01-01"

# Strategy
TREND_WINDOW_DAYS = 50
NEWS_VETO_SCORE = -0.4

# News
MAX_HEADLINES_PER_STOCK = 8
MAX_HEADLINE_AGE_DAYS = 3
NEWS_MODEL = "claude-haiku-5-5"

# Copy Berkshire
BERKSHIRE_CIK = "0001067983"
BERKSHIRE_FILINGS_SINCE = "2017-08-01"
SEC_USER_AGENT = "trend-bot droraizik@users.noreply.github.com"
FIGI_BATCH_SIZE = 10
FIGI_PAUSE_SECONDS = 3
BERKSHIRE_TOP_HOLDINGS_SHOWN = 10

# Simulation
STARTING_CASH = 10_000
COST_PER_TRADE_PCT = 0.1

# Page
SPARKLINE_DAYS = 120

# Paths
TEMPLATE_PATH = "dashboard_template.html"
PAGE_PATH = "docs/index.html"
DECISION_LOG_PATH = "docs/data/decision_log.csv"
BERKSHIRE_CACHE_PATH = "docs/data/berkshire_13f.json"
CUSIP_CACHE_PATH = "docs/data/cusip_tickers.json"


def main():
    tickers = list(TECH_STOCKS)
    prices = download_closing_prices(tickers + [BENCHMARK, SP500], START_DATE, TREND_WINDOW_DAYS * 2)
    trend_weights = decide_trend_positions(prices[tickers], TREND_WINDOW_DAYS)
    berkshire = load_berkshire_filings(BERKSHIRE_CIK, BERKSHIRE_FILINGS_SINCE, SEC_USER_AGENT, BERKSHIRE_CACHE_PATH,
                                       CUSIP_CACHE_PATH, FIGI_BATCH_SIZE, FIGI_PAUSE_SECONDS)
    berkshire_weights, berkshire_prices = build_berkshire_copy(berkshire["filings"], START_DATE, TREND_WINDOW_DAYS * 2)
    news = gather_news(TECH_STOCKS, MAX_HEADLINES_PER_STOCK, MAX_HEADLINE_AGE_DAYS, NEWS_MODEL)
    todays_weights = decide_today_with_news(trend_weights.iloc[-1], news, NEWS_VETO_SCORE)
    live_weights = update_decision_log(DECISION_LOG_PATH, prices.index[-1], todays_weights)
    history = simulate_all(prices, trend_weights, berkshire_prices, berkshire_weights, tickers, BENCHMARK, SP500,
                           START_DATE, STARTING_CASH, COST_PER_TRADE_PCT)
    live = simulate_all(prices, live_weights, berkshire_prices, berkshire_weights, tickers, BENCHMARK, SP500,
                        live_weights.index.min(), STARTING_CASH, COST_PER_TRADE_PCT)
    stock_rows = build_stock_rows(prices, trend_weights, todays_weights, news, live, TECH_STOCKS, TREND_WINDOW_DAYS, SPARKLINE_DAYS)
    page_data = build_page_data(stock_rows, history, live, berkshire, prices.index[-1], BENCHMARK, STARTING_CASH,
                                TREND_WINDOW_DAYS, NEWS_VETO_SCORE, COST_PER_TRADE_PCT, BERKSHIRE_TOP_HOLDINGS_SHOWN)
    render_dashboard(TEMPLATE_PATH, PAGE_PATH, page_data)


def decide_trend_positions(stock_prices, trend_window_days):
    in_uptrend = is_above_moving_average(stock_prices, trend_window_days)
    return equal_weights_for_selected(in_uptrend)


def load_berkshire_filings(cik, since, user_agent, cache_path, cusip_cache_path, figi_batch_size, figi_pause_seconds):
    cache = read_json(cache_path, {})
    error = None
    try:
        for filing in list_13f_filings(cik, since, user_agent):
            if filing["accession"] not in cache:
                cache[filing["accession"]] = {**filing, "holdings": fetch_13f_holdings(cik, filing["accession"], user_agent)}
                print(f"Read Berkshire 13F filed {filing['filed']}")
        write_json(cache, cache_path)
    except Exception as error_details:
        # keep going with the filings already saved, so one SEC hiccup doesn't stop the daily run
        error = f"Couldn’t reach the SEC today ({error_details})"
        print(error)

    cusip_tickers = read_json(cusip_cache_path, {})
    missing = sorted({c for f in cache.values() for c in f["holdings"]} - set(cusip_tickers))
    for start in range(0, len(missing), figi_batch_size):
        cusip_tickers.update(request_figi_tickers(missing[start:start + figi_batch_size]))
        time.sleep(figi_pause_seconds)
    write_json(cusip_tickers, cusip_cache_path)

    filings = [{"filed": f["filed"], "period": f["period"],
                "weights": holdings_to_weights(f["holdings"], cusip_tickers),
                "names": holdings_names(f["holdings"], cusip_tickers)}
               for f in sorted(cache.values(), key=lambda f: f["filed"])]
    return {"filings": filings, "error": error}


def build_berkshire_copy(filings, start_date, warmup_days):
    tickers = sorted({t for f in filings for t in f["weights"]})
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    berkshire_prices = download_closing_prices(tickers, start_date, warmup_days)
    weights = pd.DataFrame([f["weights"] for f in filings], index=pd.to_datetime([f["filed"] for f in filings]))
    weights = weights[~weights.index.duplicated(keep="last")].reindex(columns=berkshire_prices.columns).fillna(0)
    # money in stocks Yahoo has no prices for is shared among the rest, in the same proportions
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
    print(f"Copy Berkshire: {len(filings)} filings, {len(berkshire_prices.columns)} of {len(tickers)} stocks have prices")
    return weights, berkshire_prices


def gather_news(stocks, max_headlines, max_age_days, model):
    scoring_enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not scoring_enabled:
        print("No ANTHROPIC_API_KEY found: headlines are shown but not scored.")
    news = {}
    for ticker, company in stocks.items():
        headlines = fetch_recent_headlines(ticker, max_headlines, max_age_days)
        scored = score_headlines(ticker, company, headlines, model) if scoring_enabled and headlines else None
        news[ticker] = attach_scores(headlines, scored)
        print(f"{ticker}: {len(headlines)} headlines, overall score {news[ticker]['overall']}")
    return news


def decide_today_with_news(latest_trend_weights, news, news_veto_score):
    in_uptrend = latest_trend_weights > 0
    keep = apply_news_veto(in_uptrend, news, news_veto_score)
    return equal_weights_for_selected(keep.to_frame().T).iloc[0]


def update_decision_log(log_path, decision_date, todays_weights):
    log = read_decision_log(log_path)
    todays_rows = pd.DataFrame({"date": decision_date, "ticker": todays_weights.index, "weight": todays_weights.values})
    # re-running on the same day replaces that day's decision instead of duplicating it
    earlier_rows = log[log["date"] != decision_date]
    log = todays_rows if earlier_rows.empty else pd.concat([earlier_rows, todays_rows], ignore_index=True)
    save_decision_log(log, log_path)
    return log.pivot(index="date", columns="ticker", values="weight")


def simulate_all(prices, bot_weights, berkshire_prices, berkshire_weights, tickers, benchmark, sp500,
                 start_date, starting_cash, cost_per_trade_pct):
    curves = {"bot": simulate_portfolio(prices[tickers], bot_weights, start_date, starting_cash, cost_per_trade_pct)}
    if not berkshire_weights.empty:
        curves["berkshire"] = simulate_portfolio(berkshire_prices, berkshire_weights, start_date, starting_cash, cost_per_trade_pct)
    curves["sp500"] = simulate_buy_and_hold(prices[sp500], start_date, starting_cash)
    curves["benchmark"] = simulate_buy_and_hold(prices[benchmark], start_date, starting_cash)
    print(f"From {pd.Timestamp(start_date):%Y-%m-%d}: " + ", ".join(f"{k} ${c.iloc[-1]:,.0f}" for k, c in curves.items()))
    return curves


def build_stock_rows(prices, trend_weights, todays_weights, news, live, stocks, trend_window_days, sparkline_days):
    moving_average = prices[list(stocks)].rolling(trend_window_days).mean()
    portfolio_value = float(live["bot"].iloc[-1])
    rows = []
    for ticker, company in stocks.items():
        weight = float(todays_weights[ticker])
        trend_up = bool(trend_weights[ticker].iloc[-1] > 0)
        rows.append({
            "ticker": ticker,
            "name": company,
            "price": round(float(prices[ticker].iloc[-1]), 2),
            "pctVsAverage": round(percent_vs(prices[ticker].iloc[-1], moving_average[ticker].iloc[-1]), 1),
            "trendUp": trend_up,
            "status": stock_status(weight, trend_up),
            "dollars": int(round(weight * portfolio_value)),
            "newsScore": news[ticker]["overall"],
            "newsSummary": news[ticker]["summary"],
            "headlines": [headline_for_page(h) for h in news[ticker]["headlines"]],
            "spark": series_to_lists(prices[ticker].iloc[-sparkline_days:]),
            "sparkAverage": series_to_lists(moving_average[ticker].iloc[-sparkline_days:])["values"],
        })
    return rows


def build_page_data(stock_rows, history, live, berkshire, last_market_day, benchmark, starting_cash,
                    trend_window_days, news_veto_score, cost_per_trade_pct, top_holdings_shown):
    latest_filing = berkshire["filings"][-1] if berkshire["filings"] else None
    top_holdings = sorted(latest_filing["weights"].items(), key=lambda item: -item[1])[:top_holdings_shown] if latest_filing else []
    return {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "lastMarketDay": last_market_day.strftime("%Y-%m-%d"),
        "benchmark": benchmark,
        "startingCash": starting_cash,
        "newsEnabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "settings": {"trendWindowDays": trend_window_days, "newsVetoScore": news_veto_score, "costPerTradePct": cost_per_trade_pct},
        "history": {name: series_to_lists(curve) for name, curve in history.items()},
        "historyStats": {name: performance_summary(curve) for name, curve in history.items()},
        "live": {name: series_to_lists(curve) for name, curve in live.items()},
        "liveStats": {name: performance_summary(curve) for name, curve in live.items()},
        "stocks": stock_rows,
        "berkshire": {
            "error": berkshire["error"],
            "filingCount": len(berkshire["filings"]),
            "latestFiled": latest_filing["filed"] if latest_filing else None,
            "latestPeriod": latest_filing["period"] if latest_filing else None,
            "holdings": [{"ticker": t, "name": latest_filing["names"].get(t, t), "weightPct": round(w * 100, 1)} for t, w in top_holdings],
            "otherPct": round(100 - sum(w for _, w in top_holdings) * 100, 1) if latest_filing else 0,
        },
    }


if __name__ == "__main__":
    main()
