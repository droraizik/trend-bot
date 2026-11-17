import os
import pandas as pd
from datetime import datetime, timezone
from Functions.data_functions import download_closing_prices, fetch_recent_headlines, read_decision_log, save_decision_log
from Functions.signal_functions import is_above_moving_average, equal_weights_for_selected, apply_news_veto, stock_status
from Functions.news_functions import score_headlines, attach_scores
from Functions.simulation_functions import simulate_portfolio, simulate_buy_and_hold
from Functions.math_functions import performance_summary, percent_vs
from Functions.page_functions import series_to_lists, headline_for_page, render_dashboard

# Market
TECH_STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
    "META": "Meta", "AVGO": "Broadcom", "AMD": "AMD", "ORCL": "Oracle", "ADBE": "Adobe",
}
BENCHMARK = "QQQ"
START_DATE = "2018-01-01"

# Strategy
TREND_WINDOW_DAYS = 50
NEWS_VETO_SCORE = -0.4

# News
MAX_HEADLINES_PER_STOCK = 8
MAX_HEADLINE_AGE_DAYS = 3
NEWS_MODEL = "claude-haiku-5-5"

# Simulation
STARTING_CASH = 10_000
COST_PER_TRADE_PCT = 0.1

# Page
SPARKLINE_DAYS = 120

# Paths
TEMPLATE_PATH = "dashboard_template.html"
PAGE_PATH = "docs/index.html"
DECISION_LOG_PATH = "docs/data/decision_log.csv"


def main():
    tickers = list(TECH_STOCKS)
    prices = download_closing_prices(tickers + [BENCHMARK], START_DATE, TREND_WINDOW_DAYS * 2)
    trend_weights = decide_trend_positions(prices[tickers], TREND_WINDOW_DAYS)
    history = simulate_history(prices, trend_weights, tickers, BENCHMARK, START_DATE, STARTING_CASH, COST_PER_TRADE_PCT)
    news = gather_news(TECH_STOCKS, MAX_HEADLINES_PER_STOCK, MAX_HEADLINE_AGE_DAYS, NEWS_MODEL)
    todays_weights = decide_today_with_news(trend_weights.iloc[-1], news, NEWS_VETO_SCORE)
    decision_log = update_decision_log(DECISION_LOG_PATH, prices.index[-1], todays_weights)
    live = simulate_live_record(prices, decision_log, tickers, BENCHMARK, STARTING_CASH, COST_PER_TRADE_PCT)
    stock_rows = build_stock_rows(prices, trend_weights, todays_weights, news, live, TECH_STOCKS, TREND_WINDOW_DAYS, SPARKLINE_DAYS)
    page_data = build_page_data(stock_rows, history, live, prices.index[-1], BENCHMARK, STARTING_CASH,
                                TREND_WINDOW_DAYS, NEWS_VETO_SCORE, COST_PER_TRADE_PCT)
    render_dashboard(TEMPLATE_PATH, PAGE_PATH, page_data)


def decide_trend_positions(stock_prices, trend_window_days):
    in_uptrend = is_above_moving_average(stock_prices, trend_window_days)
    return equal_weights_for_selected(in_uptrend)


def simulate_history(prices, trend_weights, tickers, benchmark, start_date, starting_cash, cost_per_trade_pct):
    bot = simulate_portfolio(prices[tickers], trend_weights, start_date, starting_cash, cost_per_trade_pct)
    held_benchmark = simulate_buy_and_hold(prices[benchmark], start_date, starting_cash)
    print(f"Replay since {start_date}: bot ${bot.iloc[-1]:,.0f}, {benchmark} ${held_benchmark.iloc[-1]:,.0f}")
    return {"bot": bot, "benchmark": held_benchmark}


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
    return log


def simulate_live_record(prices, decision_log, tickers, benchmark, starting_cash, cost_per_trade_pct):
    live_weights = decision_log.pivot(index="date", columns="ticker", values="weight")
    first_day = live_weights.index.min()
    bot = simulate_portfolio(prices[tickers], live_weights, first_day, starting_cash, cost_per_trade_pct)
    held_benchmark = simulate_buy_and_hold(prices[benchmark], first_day, starting_cash)
    return {"bot": bot, "benchmark": held_benchmark}


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


def build_page_data(stock_rows, history, live, last_market_day, benchmark, starting_cash,
                    trend_window_days, news_veto_score, cost_per_trade_pct):
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
    }


if __name__ == "__main__":
    main()
