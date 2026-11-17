import json
import anthropic


def score_headlines(ticker, company, headlines, model):
    try:
        prompt = build_scoring_prompt(ticker, company, headlines)
        response = anthropic.Anthropic().messages.create(model=model, max_tokens=800, messages=[{"role": "user", "content": prompt}])
        return parse_json_reply(response.content[0].text)
    except Exception as error:
        print(f"News scoring for {ticker} failed: {error}")
        return None


def build_scoring_prompt(ticker, company, headlines):
    numbered = "\n".join(f"{i + 1}. {h['title']} ({h['source']})" for i, h in enumerate(headlines))
    return (
        f"These are recent news headlines that mention {company} ({ticker}):\n{numbered}\n\n"
        "For each headline, score how good or bad it is for the stock price over the next few weeks, "
        "from -1 (very bad) to 1 (very good). Use 0 if it is neutral, old news, or not really about the company.\n"
        'Reply with JSON only, no other text: {"scores": [one number per headline, in order], '
        '"overall": one number from -1 to 1 for the news as a whole, '
        '"summary": "one short plain-English sentence that someone who does not follow stocks would understand"}'
    )


def parse_json_reply(text):
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def attach_scores(headlines, scored):
    scored = scored or {}
    for headline, score in zip(headlines, scored.get("scores", [])):
        headline["score"] = float(score)
    overall = scored.get("overall")
    return {"headlines": headlines, "overall": None if overall is None else float(overall), "summary": scored.get("summary", "")}
