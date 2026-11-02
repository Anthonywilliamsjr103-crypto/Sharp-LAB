
import os
import math
import re
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SHARP LAB V4",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background:#091319;color:#deffff}
h1,h2,h3 {color:#66f0e3!important}
[data-testid="stMetric"] {
 background:#12252e;
 border:1px solid #24555e;
 border-radius:12px;
 padding:12px
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ SHARP LAB V4")
st.caption("All-Sports Prediction & Backtesting Engine")

try:
    st.image("goku.jpeg", use_container_width=True)
except Exception:
    st.info(
        "Upload your Goku picture to GitHub "
        "as goku.jpeg to display it here."
    )

LEAGUES = {
    "NCAAM": ("basketball", "mens-college-basketball",
              "basketball_ncaab", 3.0, 11.5),
    "NCAAW": ("basketball", "womens-college-basketball",
              "basketball_wncaab", 3.0, 12.0),
    "NBA": ("basketball", "nba",
            "basketball_nba", 2.0, 12.5),
    "WNBA": ("basketball", "wnba",
             "basketball_wnba", 2.0, 12.0),
    "NFL": ("football", "nfl",
            "americanfootball_nfl", 1.7, 12.0),
    "NCAAF": ("football", "college-football",
              "americanfootball_ncaaf", 2.8, 15.0),
    "NHL": ("hockey", "nhl",
            "icehockey_nhl", 0.16, 1.7),
    "MLB": ("baseball", "mlb",
            "baseball_mlb", 0.13, 2.6),
    "MLS": ("soccer", "usa.1",
            "soccer_usa_mls", 0.23, 1.5)
}

ET = ZoneInfo("America/New_York")


def get_key():
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return os.environ.get("ODDS_API_KEY", "")


def norm(s):
    s = unicodedata.normalize(
        "NFKD", str(s)
    ).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def local_time(s):
    return datetime.fromisoformat(
        s.replace("Z", "+00:00")
    ).astimezone(ET)


@st.cache_data(ttl=3600)
def scoreboard(league, day):
    sport, code = LEAGUES[league][:2]
    url = (
        "https://site.api.espn.com/apis/"
        f"site/v2/sports/{sport}/{code}/scoreboard"
    )
    r = requests.get(
        url,
        params={
            "dates": day.strftime("%Y%m%d"),
            "limit": 500
        },
        timeout=20
    )
    r.raise_for_status()
    return r.json().get("events", [])


@st.cache_data(ttl=300)
def get_odds(league, key):
    code = LEAGUES[league][2]
    url = (
        "https://api.the-odds-api.com/v4/"
        f"sports/{code}/odds"
    )
    r = requests.get(
        url,
        params={
            "apiKey": key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american"
        },
        timeout=25
    )
    r.raise_for_status()
    return r.json()


def parse_game(event):
    comps = event.get("competitions", [])
    if not comps:
        return None

    teams = comps[0].get("competitors", [])
    home = next(
        (t for t in teams if t.get("homeAway") == "home"),
        None
    )
    away = next(
        (t for t in teams if t.get("homeAway") == "away"),
        None
    )

    if home is None or away is None:
        return None

    def points(team):
        try:
            return float(team["score"])
        except (KeyError, ValueError, TypeError):
            return None

    return {
        "home": home["team"]["displayName"],
        "away": away["team"]["displayName"],
        "hid": str(home["team"]["id"]),
        "aid": str(away["team"]["id"]),
        "hp": points(home),
        "ap": points(away),
        "final": event.get("status", {})
            .get("type", {}).get("completed", False),
        "date": event.get("date", "")
    }


def add_history(history, game):
    if game["hp"] is None or game["ap"] is None:
        return

    for team_id, scored, allowed, is_home in [
        (game["hid"], game["hp"], game["ap"], True),
        (game["aid"], game["ap"], game["hp"], False)
    ]:
        history.setdefault(team_id, []).insert(0, {
            "scored": scored,
            "allowed": allowed,
            "is_home": is_home
        })
        history[team_id] = history[team_id][:12]


def weighted_average(games, field, home_side):
    total, weights = 0.0, 0.0

    for i, game in enumerate(games):
        weight = 0.87 ** i

        if game["is_home"] == home_side:
            weight *= 1.18

        total += game[field] * weight
        weights += weight

    return total / weights


def predict(game, history, league, min_games):
    home_games = history.get(game["hid"], [])
    away_games = history.get(game["aid"], [])

    if min(len(home_games), len(away_games)) < min_games:
        return None

    all_games = [
        x
        for team_games in history.values()
        for x in team_games
    ]

    baseline = (
        sum(g["scored"] for g in all_games)
        / len(all_games)
    )

    def ratings(games, home_side):
        shrink = 0.72 * len(games) / (len(games) + 5)

        offense = baseline + shrink * (
            weighted_average(games, "scored", home_side)
            - baseline
        )
        defense = baseline + shrink * (
            weighted_average(games, "allowed", home_side)
            - baseline
        )

        return offense, defense

    ho, hd = ratings(home_games, True)
    ao, ad = ratings(away_games, False)

    home_edge, sd = LEAGUES[league][3:]

    projected_home = max(
        0, ho + ad - baseline + home_edge / 2
    )
    projected_away = max(
        0, ao + hd - baseline - home_edge / 2
    )

    margin = projected_home - projected_away
    total = projected_home + projected_away

    p_home = 0.5 * (
        1 + math.erf(margin / (sd * math.sqrt(2)))
    )

    return {
        "margin": margin,
        "total": total,
        "home_score": projected_home,
        "away_score": projected_away,
        "p_home": p_home,
        "samples": (
            f"{len(home_games)}/{len(away_games)}"
        )
    }


def market_for_game(game, events):
    for event in events:
        if (
            norm(game["home"]) != norm(
                event.get("home_team", "")
            )
            or norm(game["away"]) != norm(
                event.get("away_team", "")
            )
        ):
            continue

        try:
            if abs(
                (
                    local_time(event["commence_time"])
                    - local_time(game["date"])
                ).total_seconds()
            ) > 86400:
                continue
        except Exception:
            continue

        result = {}

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                kind = market.get("key")

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")

                    if not isinstance(price, (int, float)):
                        continue

                    if kind == "h2h":
                        if name in (game["home"], game["away"]):
                            key = "H ML" if name == game["home"] else "A ML"
                            old = result.get(key)
                            if old is None or price > old[1]:
                                result[key] = (None, price)

                    if kind == "spreads" and isinstance(
                        point, (int, float)
                    ):
                        if name in (game["home"], game["away"]):
                            key = "H Spread" if name == game["home"] else "A Spread"
                            old = result.get(key)
                            if old is None or (point, price) > old:
                                result[key] = (point, price)

                    if kind == "totals" and isinstance(
                        point, (int, float)
                    ):
                        if name in ("Over", "Under"):
                            key = name
                            old = result.get(key)
                            score = -point if name == "Over" else point
                            old_score = (
                                -old[0] if name == "Over" else old[0]
                            ) if old else None
                            if old is None or (score, price) > (
                                old_score, old[1]
                            ):
                                result[key] = (point, price)

        return result

    return {}


with st.sidebar:
    league = st.selectbox("Sport", list(LEAGUES))
    day = st.date_input(
        "Game Date",
        datetime.now(ET).date()
    )
    lookback = st.slider(
        "History Lookback (Days)",
        14, 90, 35
    )
    min_games = st.slider(
        "Minimum Games Per Team",
        2, 12, 5
    )

    st.subheader("Model Weights")
    st.write("Recent Form — 35%")
    st.write("Matchup — 25%")
    st.write("Home/Away — 20%")
    st.write("Roster/Injuries — 20%")
    st.caption(
        "These are target weights, not all active "
        "features in the current statistical engine."
    )

    if st.button("Refresh"):
        scoreboard.clear()
        get_odds.clear()
        st.rerun()


st.header(f"Predictions — {league} — {day}")

try:
    today = [
        g
        for event in scoreboard(league, day)
        if (g := parse_game(event))
    ]
except Exception as e:
    st.error(f"Schedule unavailable: {e}")
    st.stop()

history = {}
failed = 0

with st.spinner("Loading team results..."):
    for n in range(lookback, 0, -1):
        d = day - timedelta(days=n)

        try:
            events = scoreboard(league, d)
        except Exception:
            failed += 1
            continue

        for event in events:
            game = parse_game(event)

            if game and game["final"]:
                add_history(history, game)

odds = []

if get_key():
    try:
        odds = get_odds(league, get_key())
    except Exception as e:
        st.warning(
            "Sportsbook odds unavailable. "
            "Check your API subscription and key."
        )

if failed:
    st.info(f"{failed} historical dates unavailable.")

rows = []

for game in today:
    if game["final"]:
        continue

    try:
        if local_time(game["date"]) <= datetime.now(ET):
            continue
    except Exception:
        continue

    p = predict(game, history, league, min_games)
    market = market_for_game(game, odds)

    row = {
        "Game": f'{game["away"]} @ {game["home"]}',
        "Samples": p["samples"] if p else "Insufficient",
        "Model Margin": round(p["margin"], 1) if p else None,
        "Model Total": round(p["total"], 1) if p else None,
        "Home Win %": round(p["p_home"] * 100, 1) if p else None,
        "Home Spread": market.get("H Spread", (None, None))[0],
        "Away Spread": market.get("A Spread", (None, None))[0],
        "Market Total": market.get("Over", (None, None))[0],
        "Home ML": market.get("H ML", (None, None))[1],
        "Away ML": market.get("A ML", (None, None))[1],
        "Model Lean": "PASS"
    }

    if p and "H Spread" in market:
        spread = market["H Spread"][0]
        edge = p["margin"] + spread
        if edge > 2.5:
            row["Model Lean"] = f'{game["home"]} {spread:+g}'
        elif edge < -2.5 and "A Spread" in market:
            row["Model Lean"] = (
                f'{game["away"]} '
                f'{market["A Spread"][0]:+g}'
            )

    rows.append(row)

if rows:
    results = pd.DataFrame(rows)
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Export Predictions",
        results.to_csv(index=False),
        "sharp_lab_predictions.csv",
        "text/csv"
    )
else:
    st.info("No upcoming games found for the selected date.")


st.divider()
st.header("📊 Historical Backtesting")

st.caption(
    "Walk-forward testing: a game is predicted "
    "using only earlier completed games."
)

bt_days = st.slider("Backtest Days", 7, 30, 14)
warmup = st.slider("Training History (Days)", 14, 70, 35)
cutoff = st.date_input(
    "Backtest Ending Date",
    min(day, datetime.now(ET).date() - timedelta(days=1))
)

lines_file = st.file_uploader(
    "Historical Lines CSV (optional)",
    type="csv"
)

st.caption(
    "For ATS and totals, CSV columns must be: "
    "date,home,away,home_spread,total"
)

if st.button("Run Backtest", type="primary"):
    closing = {}

    if lines_file is not None:
        try:
            lines = pd.read_csv(lines_file)
            needed = {
                "date", "home", "away",
                "home_spread", "total"
            }

            if not needed.issubset(lines.columns):
                st.error("Historical CSV has missing columns.")
                st.stop()

            for _, r in lines.iterrows():
                closing[(
                    str(r["date"])[:10],
                    norm(r["home"]),
                    norm(r["away"])
                )] = (
                    float(r["home_spread"]),
                    float(r["total"])
                )
        except Exception as e:
            st.error(f"Could not read lines CSV: {e}")
            st.stop()

    start = cutoff - timedelta(days=bt_days - 1)
    earliest = start - timedelta(days=warmup)

    past = {}
    tested = []
    skipped = 0

    total_days = (cutoff - earliest).days + 1
    progress = st.progress(0)

    for i in range(total_days):
        test_date = earliest + timedelta(days=i)

        try:
            finished = [
                g
                for ev in scoreboard(league, test_date)
                if (g := parse_game(ev))
                and g["final"]
                and g["hp"] is not None
                and g["ap"] is not None
            ]
        except Exception:
            finished = []

        # Predictions happen before adding this day's scores.
        if test_date >= start:
            for game in finished:
                prediction = predict(
                    game, past, league, min_games
                )

                if prediction is None:
                    skipped += 1
                    continue

                actual_margin = game["hp"] - game["ap"]
                actual_total = game["hp"] + game["ap"]

                row = {
                    "Date": test_date.isoformat(),
                    "Game": (
                        f'{game["away"]} @ {game["home"]}'
                    ),
                    "Pred Margin": prediction["margin"],
                    "Actual Margin": actual_margin,
                    "Pred Total": prediction["total"],
                    "Actual Total": actual_total,
                    "Winner Correct": None,
                    "ATS Correct": None,
                    "Total Correct": None,
                    "Margin Error": abs(
                        prediction["margin"] - actual_margin
                    ),
                    "Total Error": abs(
                        prediction["total"] - actual_total
                    ),
                    "Brier": None
                }

                if actual_margin != 0:
                    row["Winner Correct"] = int(
                        (prediction["margin"] > 0)
                        == (actual_margin > 0)
                    )
                    row["Brier"] = (
                        prediction["p_home"]
                        - int(actual_margin > 0)
                    ) ** 2

                key = (
                    test_date.isoformat(),
                    norm(game["home"]),
                    norm(game["away"])
                )

                if key in closing:
                    spread, total_line = closing[key]
                    model_spread = prediction["margin"] + spread
                    actual_spread = actual_margin + spread

                    if model_spread != 0 and actual_spread != 0:
                        row["ATS Correct"] = int(
                            (model_spread > 0)
                            == (actual_spread > 0)
                        )

                    model_total = prediction["total"] - total_line
                    actual_total_diff = actual_total - total_line

                    if model_total != 0 and actual_total_diff != 0:
                        row["Total Correct"] = int(
                            (model_total > 0)
                            == (actual_total_diff > 0)
                        )

                tested.append(row)

        for game in finished:
            add_history(past, game)

        progress.progress((i + 1) / total_days)

    if not tested:
        st.warning(
            "No eligible historical games. "
            "Try fewer minimum samples or a longer training window."
        )
    else:
        frame = pd.DataFrame(tested)

        wins = frame["Winner Correct"].dropna()
        ats = frame["ATS Correct"].dropna()
        totals = frame["Total Correct"].dropna()

        a, b, c, d = st.columns(4)

        a.metric("Games Tested", len(frame))
        b.metric(
            "Winner Accuracy",
            f"{wins.mean():.1%}" if len(wins) else "N/A"
        )
        c.metric(
            "Margin MAE",
            f'{frame["Margin Error"].mean():.2f}'
        )
        d.metric(
            "Total MAE",
            f'{frame["Total Error"].mean():.2f}'
        )

        st.write(
            "ATS Accuracy:",
            f"{ats.mean():.1%}" if len(ats) else "Needs historical lines"
        )
        st.write(
            "Totals Accuracy:",
            f"{totals.mean():.1%}" if len(totals) else "Needs historical lines"
        )
        st.write(
            "Brier Score:",
            round(frame["Brier"].mean(), 3)
        )
        st.caption(f"{skipped} games skipped for insufficient history.")

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Backtest Results",
            frame.to_csv(index=False),
            "sharp_lab_backtest.csv",
            "text/csv"
        )

st.warning(
    "Experimental predictions only. The engine currently "
    "uses historical scoring and home advantage. "
    "Roster injuries, opponent-adjusted efficiency, "
    "pitchers, goalies, and weather are not automated. "
    "Win probabilities are not calibrated. "
    "Historical accuracy does not guarantee future results."
)
