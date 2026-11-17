def total_return_pct(equity):
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


def max_drawdown_pct(equity):
    return (equity / equity.cummax() - 1).min() * 100


def percent_vs(value, reference):
    return (float(value) / float(reference) - 1) * 100


def performance_summary(equity):
    return {
        "final": int(round(float(equity.iloc[-1]))),
        "returnPct": round(float(total_return_pct(equity)), 1),
        "maxDropPct": round(float(max_drawdown_pct(equity)), 1),
    }
