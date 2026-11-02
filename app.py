
import math
import os
import re
import statistics
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SHARP LAB Predictions",
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

st.title("⚡ SHARP LAB — PREDICTION ENGINE")
st.caption(
    "Independent performance projections | "
    "Spreads | Totals | Moneylines"
)

LEAGUES = {
    "NCAAM": ("basketball", "mens-college-basketball",
              "basketball_ncaab", 11.5, 3.0, 145),
    "NCAAW": ("basketball", "womens-college-basketball",
              "basketball_wncaab", 12.0, 3.0, 135),
    "NBA": ("basketball", "nba",
            "basketball_nba", 12.5, 2.0, 227),
    "WNBA": ("basketball", "wnba",
             "basketball_wnba", 12.0, 2.0, 165),
    "NFL": ("football", "nfl",
            "americanfootball_nfl", 12.0, 1.7, 44),
    "NCAAF": ("football", "college-football",
              "americanfootball_ncaaf", 15.0, 2.8, 53),
    "NHL": ("hockey", "nhl",
            "icehockey_nhl", 1.7, 0.16, 6.1),
    "MLB": ("baseball", "mlb",
            "baseball_mlb", 2.6, 0.13, 8.8),
    "MLS": ("soccer", "usa.1",
            "soccer_usa_mls", 1.5, 0.23, 2.9)
}

ET = ZoneInfo("America/New_York")


def get_key():
    try:
        return (
            st.secrets.get("ODDS_API_KEY", "")
            or os.environ.get("ODDS_API_KEY", "")
        )
    except Exception:
        return os.environ.get("ODDS_API_KEY", "")


def normalize(name):
    name = unicodedata.normalize(
        "NFKD", str(name)
    ).encode("ascii", "ignore").decode().lower()

    return re.sub(
        r"[^a-z0-9]", "",
        name.replace("&", "and")
    )


def local_time(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(ET)


@st.cache_data(ttl=3600)
def get_schedule(league, game_date):
    sport, code = LEAGUES[league][:2]

    url = (
        "https://site.api.espn.com/apis/"
        f"site/v2/sports/{sport}/{code}/scoreboard"
    )

    response = requests.get(
        url,
        params={
            "dates": game_date.strftime("%Y%m%d"),
            "limit": 500
        },
        timeout=18
    )

    response.raise_for_status()

    return response.json().get("events", [])


@st.cache_data(ttl=300)
def get_odds(league, api_key):
    sport_key = LEAGUES[league][2]

    url = (
        "https://api.the-odds-api.com/v4/"
        f"sports/{sport_key}/odds"
    )

    response = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american"
        },
        timeout=25
    )

    response.raise_for_status()
    return response.json()


def parse_game(event):
    competitions = event.get("competitions", [])

    if not competitions:
        return None

    teams = competitions[0].get("competitors", [])

    home = next(
        (t for t in teams if t.get("homeAway") == "home"),
        None
    )

    away = next(
        (t for t in teams if t.get("homeAway") == "away"),
        None
    )

    if not home or not away:
        return None

    status = event.get(
        "status", {}
    ).get("type", {})

    def score(team):
        try:
            return float(team["score"])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "home": home.get("team", {}).get(
            "displayName", ""
        ),
        "away": away.get("team", {}).get(
            "displayName", ""
        ),
        "home_id": str(
            home.get("team", {}).get("id", "")
        ),
        "away_id": str(
            away.get("team", {}).get("id", "")
        ),
        "home_pts": score(home),
        "away_pts": score(away),
        "final": status.get("completed", False),
        "date": event.get("date", "")
    }


def weighted_mean(games, stat):
    if not games:
        return None

    values = []

    for index, game in enumerate(games):
        weight = 0.87 ** index

        if game.get("at_target_venue", False):
            weight *= 1.18

        values.append(
            (game[stat], weight)
        )

    return (
        sum(value * weight for value, weight in values)
        / sum(weight for _, weight in values)
    )


def make_prediction(
    league, home, away, history, minimum_games
):
    sd, home_edge, base = LEAGUES[league][3:]

    home_games = history.get(
        home["home_id"], []
    )[:12]

    away_games = history.get(
        away["away_id"], []
    )[:12]

    if min(
        len(home_games), len(away_games)
    ) < minimum_games:

        return None, (
            f"Insufficient history: "
            f"{len(home_games)} home-team games, "
            f"{len(away_games)} away-team games."
        )

    home_games = [
        dict(g, at_target_venue=g["is_home"])
        for g in home_games
    ]

    away_games = [
        dict(g, at_target_venue=not g["is_home"])
        for g in away_games
    ]

    all_games = [
        g
        for games in history.values()
        for g in games
    ]

    league_average = (
        statistics.mean(
            g["scored"] for g in all_games
        )
        if all_games else base / 2
    )

    baseline = (
        0.6 * league_average
        + 0.4 * (base / 2)
    )

    home_weight = (
        min(1.0, len(home_games) / (len(home_games) + 5))
        * 0.72
    )

    away_weight = (
        min(1.0, len(away_games) / (len(away_games) + 5))
        * 0.72
    )

    home_offense = baseline + home_weight * (
        weighted_mean(home_games, "scored") - baseline
    )

    home_defense = baseline + home_weight * (
        weighted_mean(home_games, "allowed") - baseline
    )

    away_offense = baseline + away_weight * (
        weighted_mean(away_games, "scored") - baseline
    )

    away_defense = baseline + away_weight * (
        weighted_mean(away_games, "allowed") - baseline
    )

    projected_home = max(
        0,
        baseline
        + (home_offense - baseline)
        + (away_defense - baseline)
        + home_edge / 2
    )

    projected_away = max(
        0,
        baseline
        + (away_offense - baseline)
        + (home_defense - baseline)
        - home_edge / 2
    )

    margin = projected_home - projected_away
    total = projected_home + projected_away

    home_win = 0.5 * (
        1 + math.erf(
            margin / (sd * math.sqrt(2))
        )
    )

    return {
        "margin": margin,
        "total": total,
        "home_score": projected_home,
        "away_score": projected_away,
        "home_win": home_win,
        "sd": sd,
        "sample": (
            f"{len(home_games)}/{len(away_games)}"
        )
    }, None


def best_market_lines(event):
    results = {
        "home_ml": None,
        "away_ml": None,
        "home_spread": None,
        "away_spread": None,
        "over": None,
        "under": None
    }

    if not event:
        return results

    home = event["home_team"]
    away = event["away_team"]

    for bookmaker in event.get("bookmakers", []):

        book = bookmaker.get("title", "")

        for market in bookmaker.get("markets", []):

            kind = market.get("key")

            for outcome in market.get("outcomes", []):

                name = outcome.get("name")
                price = outcome.get("price")
                point = outcome.get("point")

                if not isinstance(price, (int, float)):
                    continue

                if kind == "h2h" and name in (home, away):

                    field = (
                        "home_ml" if name == home
                        else "away_ml"
                    )

                    choice = (price, book)

                    if (
                        results[field] is None
                        or price > results[field][0]
                    ):
                        results[field] = choice

                if (
                    kind == "spreads"
                    and name in (home, away)
                    and isinstance(point, (int, float))
                ):

                    field = (
                        "home_spread" if name == home
                        else "away_spread"
                    )

                    choice = (point, price, book)
                    old = results[field]

                    if (
                        old is None
                        or (point, price) > (old[0], old[1])
                    ):
                        results[field] = choice

                if (
                    kind == "totals"
                    and name in ("Over", "Under")
                    and isinstance(point, (int, float))
                ):

                    field = (
                        "over" if name == "Over"
                        else "under"
                    )

                    choice = (point, price, book)
                    old = results[field]

                    score = (
                        -point if field == "over"
                        else point
                    )

                    if old is None:
                        results[field] = choice
                    else:
                        old_score = (
                            -old[0] if field == "over"
                            else old[0]
                        )

                        if (score, price) > (
                            old_score, old[1]
                        ):
                            results[field] = choice

    return results


def match_market(game, odds):
    for event in odds:

        same_home = (
            normalize(game["home"])
            == normalize(event.get("home_team", ""))
        )

        same_away = (
            normalize(game["away"])
            == normalize(event.get("away_team", ""))
        )

        if same_home and same_away:
            try:
                difference = abs(
                    (
                        local_time(event["commence_time"])
                        - local_time(game["date"])
                    ).total_seconds()
                )

                if difference < 86400:
                    return event

            except Exception:
                pass

    return None


def format_odds(value):
    if value is None:
        return "—"

    return (
        f"{value:+.0f}" if value > 0
        else str(int(value))
    )


with st.sidebar:
    st.header("⚙️ SHARP LAB")

    league = st.selectbox(
        "Sport",
        list(LEAGUES)
    )

    day = st.date_input(
        "Game Date",
        value=datetime.now(ET).date()
    )

    lookback = st.slider(
        "Historical Lookback (Days)",
        14, 90, 35, 7
    )

    minimum_games = st.slider(
        "Minimum Games Per Team",
        2, 12, 5
    )

    st.divider()

    st.subheader("Model Framework")

    st.write("Recent Form: 35%")
    st.write("Matchups: 25%")
    st.write("Home/Away: 20%")
    st.write("Roster/Injuries: 20%")

    st.caption(
        "These are proposed weights. "
        "Missing data is not fabricated."
    )

    if st.button("🔄 Refresh Data"):
        get_schedule.clear()
        get_odds.clear()
        st.rerun()


st.header(
    f"🏆 {league} Predictions — {day}"
)

history = {}
seen = set()
failed_days = 0

with st.spinner(
    "Analyzing team history and today's games..."
):

    try:
        today = []

        for event in get_schedule(league, day):
            parsed = parse_game(event)

            if parsed:
                today.append(parsed)

    except Exception as error:
        st.error(
            f"Schedule unavailable: {error}"
        )
        st.stop()

    for days_back in range(1, lookback + 1):

        historical_date = (
            day - timedelta(days=days_back)
        )

        try:
            events = get_schedule(
                league, historical_date
            )
        except Exception:
            failed_days += 1
            continue

        for event in events:

            game = parse_game(event)

            if not game or not game["final"]:
                continue

            if (
                game["home_pts"] is None
                or game["away_pts"] is None
            ):
                continue

            identifier = (
                game["date"],
                game["home_id"],
                game["away_id"]
            )

            if identifier in seen:
                continue

            seen.add(identifier)

            history.setdefault(
                game["home_id"], []
            ).append({
                "scored": game["home_pts"],
                "allowed": game["away_pts"],
                "is_home": True
            })

            history.setdefault(
                game["away_id"], []
            ).append({
                "scored": game["away_pts"],
                "allowed": game["home_pts"],
                "is_home": False
            })

    odds = []
    odds_error = ""

    if get_key():
        try:
            odds = get_odds(
                league, get_key()
            )
        except Exception:
            odds_error = (
                "Odds feed unavailable. Check "
                "API key, credits, and coverage."
            )
    else:
        odds_error = (
            "No Odds API key found in Streamlit Secrets."
        )


if failed_days:
    st.info(
        f"Missing historical data for "
        f"{failed_days} days."
    )

if odds_error:
    st.info(odds_error)

rows = []

for game in today:

    if game["final"]:
        continue

    try:
        tip_time = local_time(game["date"])

        if tip_time <= datetime.now(ET):
            continue

    except Exception:
        continue

    prediction, reason = make_prediction(
        league,
        game,
        game,
        history,
        minimum_games
    )

    market = best_market_lines(
        match_market(game, odds)
    )

    row = {
        "Game": (
            f'{game["away"]} @ {game["home"]}'
        ),
        "Time ET": tip_time.strftime("%I:%M %p"),
        "Model Home Margin": None,
        "Model Total": None,
        "Home Win %": None,
        "Home Spread": "—",
        "Away Spread": "—",
        "Market Total": None,
        "Home ML": "—",
        "Away ML": "—",
        "Best Lean": "PASS",
        "Samples": "",
        "Reason": reason or ""
    }

    if prediction:
        row["Model Home Margin"] = round(
            prediction["margin"], 1
        )

        row["Model Total"] = round(
            prediction["total"], 1
        )

        row["Home Win %"] = round(
            prediction["home_win"] * 100, 1
        )

        row["Samples"] = prediction["sample"]

    if market["home_spread"]:
        point, price, book = market["home_spread"]

        row["Home Spread"] = (
            f"{point:+g} ({format_odds(price)})"
        )

    if market["away_spread"]:
        point, price, book = market["away_spread"]

        row["Away Spread"] = (
            f"{point:+g} ({format_odds(price)})"
        )

    if market["over"]:
        row["Market Total"] = market["over"][0]

    if market["home_ml"]:
        row["Home ML"] = format_odds(
            market["home_ml"][0]
        )

    if market["away_ml"]:
        row["Away ML"] = format_odds(
            market["away_ml"][0]
        )

    if prediction:
        suggestions = []

        for side in ("home", "away"):

            quote = market[f"{side}_spread"]

            if not quote:
                continue

            point = quote[0]

            edge = (
                prediction["margin"] + point
            ) * (1 if side == "home" else -1)

            threshold = (
                2.5 * prediction["sd"] / 12
            )

            if edge >= threshold:
                suggestions.append((
                    edge / prediction["sd"],
                    f'{game[side]} {point:+g}'
                ))

        if market["over"]:

            difference = (
                prediction["total"]
                - market["over"][0]
            )

            threshold = max(
                4.5, prediction["sd"] * 0.5
            )

            if abs(difference) >= threshold:

                selection = (
                    "OVER" if difference > 0
                    else "UNDER"
                )

                suggestions.append((
                    abs(difference)
                    / (prediction["sd"] * 2),
                    selection
                ))

        if suggestions:

            suggestions.sort(reverse=True)

            row["Best Lean"] = (
                suggestions[0][1]
                + " (Exploratory)"
            )

            row["Reason"] = (
                "Historical model differs "
                "from sportsbook line."
            )

        else:
            row["Reason"] = (
                "No sufficient projected edge "
                "or missing odds."
            )

    rows.append(row)


if not rows:

    st.info(
        "No unstarted games found for this date. "
        "Try another day."
    )

else:

    results = pd.DataFrame(rows)

    c1, c2 = st.columns(2)

    c1.metric(
        "Upcoming Games",
        len(results)
    )

    c2.metric(
        "Games With Projections",
        int(
            results[
                "Model Home Margin"
            ].notna().sum()
        )
    )

    st.subheader("📊 Today's Predictions")

    st.dataframe(
        results.drop(columns=["Reason"]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Predictions CSV",
        results.to_csv(index=False),
        file_name=(
            f"sharp_lab_{league}_{day}.csv"
        ),
        mime="text/csv"
    )

    with st.expander("Model Notes / PASS Reasons"):

        st.dataframe(
            results[
                [
                    "Game",
                    "Best Lean",
                    "Samples",
                    "Reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


st.warning(
    "This model is experimental and not "
    "historically calibrated. Win probabilities "
    "and spread/total leans are estimates, "
    "not verified betting edges. "
    "Injuries, lineups, opponent-adjusted "
    "efficiency, pitchers, goalies, and "
    "weather are not yet automatically included. "
    "Early-season predictions can be especially "
    "unreliable due to limited game history."
)
