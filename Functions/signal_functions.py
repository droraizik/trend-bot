import pandas as pd


def is_above_moving_average(prices, window_days):
    return prices > prices.rolling(window_days).mean()


def equal_weights_for_selected(selected):
    count = selected.sum(axis=1)
    return selected.astype(float).div(count.where(count > 0), axis=0).fillna(0)


def apply_news_veto(in_uptrend, news, veto_score):
    bad_news = pd.Series({t: news[t]["overall"] is not None and news[t]["overall"] <= veto_score for t in in_uptrend.index})
    return in_uptrend & ~bad_news


def stock_status(weight, trend_up):
    if weight > 0:
        return "hold"
    return "vetoed" if trend_up else "out"
