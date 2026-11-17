def simulate_portfolio(stock_prices, target_weights, start_date, starting_cash, cost_per_trade_pct):
    stock_prices = stock_prices.loc[start_date:]
    target_weights = target_weights.reindex(index=stock_prices.index, columns=stock_prices.columns).ffill().fillna(0)
    # decided at a day's close, held from the next day, so the bot never uses future prices
    held = target_weights.shift(1).fillna(0)
    gross_returns = (held * stock_prices.pct_change().fillna(0)).sum(axis=1)
    traded_fraction = held.diff().abs().sum(axis=1).fillna(0)
    return starting_cash * (1 + gross_returns - traded_fraction * cost_per_trade_pct / 100).cumprod()


def simulate_buy_and_hold(prices, start_date, starting_cash):
    prices = prices.loc[start_date:]
    return starting_cash * prices / prices.iloc[0]
