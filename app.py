
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SHARP LAB — All Sports",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #081019;
    color: #e5ffff;
}
h1, h2, h3 {
    color: #52eddd !important;
}
div[data-testid="stMetric"] {
    background: #10252a;
    border: 1px solid #176b69;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ SHARP LAB — ALL SPORTS")
st.caption(
    "Automatic sportsbook odds • Spreads • "
    "Moneylines • Totals • Underdogs"
)

SPORTS = {
    "NCAAM": "basketball_ncaab",
    "NCAAW": "basketball_wncaab",
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
    "MLS": "soccer_usa_mls"
}

MARKETS = {
    "h2h": "Moneyline",
    "spreads": "Spread",
    "totals": "Total"
}

ET = ZoneInfo("America/New_York")


def get_api_key():
    try:
        return st.secrets.get(
            "ODDS_API_KEY", ""
        ) or os.environ.get("ODDS_API_KEY", "")
    except Exception:
        return os.environ.get("ODDS_API_KEY", "")


@st.cache_data(ttl=3600)
def get_available_sports(key):
    url = "https://api.the-odds-api.com/v4/sports"

    response = requests.get(
        url,
        params={"apiKey": key},
        timeout=20
    )

    response.raise_for_status()

    return {
        sport["key"]
        for sport in response.json()
    }


@st.cache_data(ttl=300)
def get_odds(sport_key, key, region):
    url = (
        "https://api.the-odds-api.com/v4/sports/"
        f"{sport_key}/odds"
    )

    params = {
        "apiKey": key,
        "regions": region,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }

    response = requests.get(
        url,
        params=params,
        timeout=25
    )

    if not response.ok:
        try:
            message = response.json().get(
                "message", response.text[:200]
            )
        except Exception:
            message = response.text[:200]

        raise RuntimeError(
            f"HTTP {response.status_code}: {message}"
        )

    remaining = response.headers.get(
        "x-requests-remaining", "Unknown"
    )

    return response.json(), remaining


def local_time(timestamp):
    return datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    ).astimezone(ET)


def implied_probability(odds):
    odds = float(odds)

    if odds == 0:
        return None

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def calculate_ev(probability, odds):
    odds = float(odds)

    if odds == 0:
        return None

    profit = (
        odds / 100 if odds > 0
        else 100 / abs(odds)
    )

    return (
        probability * profit
        - (1 - probability)
    ) * 100


def process_events(events, sport, selected_date):
    rows = []

    for event in events:
        try:
            start = local_time(
                event["commence_time"]
            )
        except (KeyError, ValueError, TypeError):
            continue

        if start.date() != selected_date:
            continue

        home = event.get("home_team", "Home")
        away = event.get("away_team", "Away")

        for bookmaker in event.get(
            "bookmakers", []
        ):
            book_name = bookmaker.get(
                "title", "Unknown"
            )

            for market in bookmaker.get(
                "markets", []
            ):
                market_key = market.get("key")

                if market_key not in MARKETS:
                    continue

                for outcome in market.get(
                    "outcomes", []
                ):
                    price = outcome.get("price")

                    if price is None:
                        continue

                    rows.append({
                        "Sport": sport,
                        "Game": f"{away} @ {home}",
                        "Start ET": start.strftime(
                            "%Y-%m-%d %I:%M %p"
                        ),
                        "Bookmaker": book_name,
                        "Market": MARKETS[market_key],
                        "Selection": outcome.get(
                            "name", ""
                        ),
                        "Line": outcome.get("point"),
                        "Odds": price
                    })

    return rows


def moneyline_comparisons(data):
    moneylines = data[
        data["Market"] == "Moneyline"
    ]

    fair_rows = []

    for (
        sport, game, book
    ), group in moneylines.groupby(
        ["Sport", "Game", "Bookmaker"]
    ):
        # Only two-outcome markets.
        # Soccer may include a draw.
        if len(group) != 2:
            continue

        probabilities = []

        for _, item in group.iterrows():
            probability = implied_probability(
                item["Odds"]
            )

            if probability is None:
                break

            probabilities.append(
                (item["Selection"], probability)
            )

        if len(probabilities) != 2:
            continue

        total = sum(
            p for _, p in probabilities
        )

        if total <= 0:
            continue

        for selection, probability in probabilities:
            fair_rows.append({
                "Sport": sport,
                "Game": game,
                "Bookmaker": book,
                "Selection": selection,
                "Fair probability": probability / total
            })

    if not fair_rows:
        return pd.DataFrame()

    fair = pd.DataFrame(fair_rows)
    results = []

    grouped = fair.groupby(
        ["Sport", "Game", "Selection"]
    )

    for (
        sport, game, selection
    ), group in grouped:

        for _, row in group.iterrows():
            book = row["Bookmaker"]

            other_books = group[
                group["Bookmaker"] != book
            ]

            if len(other_books) < 2:
                continue

            reference = float(
                other_books[
                    "Fair probability"
                ].median()
            )

            matching = moneylines[
                (moneylines["Sport"] == sport)
                & (moneylines["Game"] == game)
                & (moneylines["Bookmaker"] == book)
                & (moneylines["Selection"] == selection)
            ]

            if matching.empty:
                continue

            odds = matching.iloc[0]["Odds"]

            ev = calculate_ev(
                reference, odds
            )

            results.append({
                "Sport": sport,
                "Game": game,
                "Selection": selection,
                "Bookmaker": book,
                "Odds": odds,
                "Other-books fair %": round(
                    reference * 100, 1
                ),
                "Market comparison EV %": round(
                    ev, 2
                )
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        results
    ).sort_values(
        "Market comparison EV %",
        ascending=False
    )


with st.sidebar:
    st.header("⚙️ SHARP LAB Settings")

    selected_sport = st.selectbox(
        "Select Sport",
        ["ALL SPORTS"] + list(SPORTS.keys())
    )

    selected_date = st.date_input(
        "Game Date",
        value=datetime.now(ET).date()
    )

    region = st.selectbox(
        "Sportsbook Region",
        ["us", "us2", "uk", "eu", "au"]
    )

    if st.button("🔄 Refresh Odds"):
        get_odds.clear()
        st.rerun()

    st.divider()

    st.header("Model Weights")

    recent_weight = st.slider(
        "Recent Form %",
        0, 100, 35
    )

    matchup_weight = st.slider(
        "Matchup Strength %",
        0, 100, 25
    )

    home_weight = st.slider(
        "Home/Away %",
        0, 100, 20
    )

    roster_weight = st.slider(
        "Roster / Injuries %",
        0, 100, 20
    )

    st.header("Home/Away Splits")

    win_weight = st.slider(
        "Win Percentage Share",
        0, 100, 25
    )

    ats_weight = st.slider(
        "ATS Percentage Share",
        0, 100, 30
    )

    margin_weight = st.slider(
        "Scoring Margin Share",
        0, 100, 25
    )

    schedule_weight = st.slider(
        "Strength of Schedule Share",
        0, 100, 20
    )

    if (
        recent_weight
        + matchup_weight
        + home_weight
        + roster_weight
    ) != 100:
        st.warning(
            "Main weights should total 100%."
        )

    if (
        win_weight
        + ats_weight
        + margin_weight
        + schedule_weight
    ) != 100:
        st.warning(
            "Home/away weights should total 100%."
        )

st.subheader(
    f"Sportsbook Board — {selected_date}"
)

key = get_api_key()

if not key:
    st.error(
        "Missing ODDS_API_KEY. Add your key "
        "to Streamlit App Settings → Secrets."
    )

    st.code(
        'ODDS_API_KEY = "YOUR_API_KEY"',
        language="toml"
    )

    st.stop()

try:
    available_sports = get_available_sports(
        key
    )
except Exception as error:
    st.error(
        f"Unable to connect to odds provider: {error}"
    )
    st.stop()

if selected_sport == "ALL SPORTS":
    selected_leagues = list(SPORTS.items())
else:
    selected_leagues = [
        (
            selected_sport,
            SPORTS[selected_sport]
        )
    ]

all_rows = []
errors = []
remaining = "Unknown"

for league, sport_key in selected_leagues:
    if sport_key not in available_sports:
        errors.append(
            f"{league}: unavailable from provider"
        )
        continue

    try:
        events, remaining = get_odds(
            sport_key, key, region
        )

        new_rows = process_events(
            events,
            league,
            selected_date
        )

        all_rows.extend(new_rows)

    except Exception as error:
        errors.append(
            f"{league}: {error}"
        )

if errors:
    with st.expander(
        f"Coverage notes ({len(errors)})"
    ):
        for error in errors:
            st.write(error)

st.caption(
    f"API requests remaining: {remaining}. "
    "Prices are provider snapshots."
)

if not all_rows:
    st.info(
        "No available betting markets were "
        "returned for this date. Check the "
        "league, date, API plan, or coverage."
    )
    st.stop()

data = pd.DataFrame(all_rows)

games = data[
    ["Sport", "Game", "Start ET"]
].drop_duplicates()

st.success(
    f"{len(games)} games loaded automatically!"
)

st.dataframe(
    games,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download All Sportsbook Odds",
    data.to_csv(index=False),
    file_name="sharp_lab_odds.csv",
    mime="text/csv"
)

st.divider()

st.header("🏀 Matchup Odds Analyzer")

game_options = (
    games["Sport"].astype(str)
    + " | "
    + games["Game"].astype(str)
).tolist()

selected_game = st.selectbox(
    "Choose a Matchup",
    game_options
)

sport, game_name = selected_game.split(
    " | ", 1
)

game_data = data[
    (data["Sport"] == sport)
    & (data["Game"] == game_name)
]

selected_markets = st.multiselect(
    "Choose Betting Markets",
    ["Moneyline", "Spread", "Total"],
    default=[
        "Moneyline",
        "Spread",
        "Total"
    ]
)

visible = game_data[
    game_data["Market"].isin(
        selected_markets
    )
]

st.dataframe(
    visible[
        [
            "Bookmaker",
            "Market",
            "Selection",
            "Line",
            "Odds"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.divider()

st.header("💰 Underdog and Price Analysis")

ranked = moneyline_comparisons(data)

if ranked.empty:
    st.info(
        "Not enough independent sportsbook "
        "quotes to calculate comparisons."
    )
else:
    underdogs_only = st.checkbox(
        "Show Plus-Money Underdogs Only"
    )

    if underdogs_only:
        ranked = ranked[
            ranked["Odds"] > 0
        ]

    st.dataframe(
        ranked.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Market Comparisons",
        ranked.to_csv(index=False),
        file_name="sharp_lab_comparisons.csv",
        mime="text/csv"
    )

st.divider()

st.warning(
    "SHARP LAB automatically retrieves "
    "available sportsbook prices. Its "
    "market comparisons are not independent "
    "predictions or guaranteed betting edges. "
    "Roster strength, injuries, recent form, "
    "ATS records and predictive probabilities "
    "are not yet automatically modeled. "
    "The weight settings above are saved "
    "as on-screen inputs but do not currently "
    "change the odds comparisons."
)
