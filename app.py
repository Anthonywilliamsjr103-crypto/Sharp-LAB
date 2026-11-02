
import streamlit as st
import requests
import math
from datetime import date

st.set_page_config(page_title="SHARP LAB", page_icon="🏀", layout="wide")

st.markdown("""
<style>
.stApp {background: #080e14; color: #e5ffff;}
h1,h2,h3 {color: #52eddd !important;}
div[data-testid="stMetric"] {
    background: #10252a;
    border: 1px solid #176b69;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ SHARP LAB — ALL SPORTS")
st.caption("Multi-sport matchup analytics • Schedule feed • Betting models")

LEAGUES = {
    "NCAAM": ("basketball", "mens-college-basketball"),
    "NCAAW": ("basketball", "womens-college-basketball"),
    "NBA": ("basketball", "nba"),
    "WNBA": ("basketball", "wnba"),
    "NFL": ("football", "nfl"),
    "NCAAF": ("football", "college-football"),
    "NHL": ("hockey", "nhl"),
    "MLB": ("baseball", "mlb"),
    "SOCCER": ("soccer", "usa.1"),
}

HOME_EDGE = {
    "NCAAM": 3.2, "NCAAW": 3.2,
    "NBA": 2.2, "WNBA": 2.0,
    "NFL": 1.6, "NCAAF": 2.8,
    "NHL": 0.18, "MLB": 0.12, "SOCCER": 0.25
}
MARGIN_SD = {
    "NCAAM": 11.5, "NCAAW": 12,
    "NBA": 12.5, "WNBA": 12,
    "NFL": 12, "NCAAF": 15,
    "NHL": 1.6, "MLB": 2.5, "SOCCER": 1.3
}

@st.cache_data(ttl=120)
def get_games(league, day):
    sport, code = LEAGUES[league]
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{sport}/{code}/scoreboard"
    )
    response = requests.get(
        url,
        params={"dates": day.strftime("%Y%m%d"), "limit": 500},
        timeout=18,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    return response.json().get("events", [])

def normal_cdf(value):
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))

def american_to_decimal(odds):
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

def value_ev(probability, odds):
    decimal = american_to_decimal(odds)
    return (probability * decimal - 1) * 100

with st.sidebar:
    st.header("Model settings")
    league = st.selectbox("Sport", list(LEAGUES))
    day = st.date_input("Game date", value=date.today())
    st.markdown("### Model weights")
    recent_w = st.number_input("Recent form %", 0, 100, 35)
    matchup_w = st.number_input("Matchups %", 0, 100, 25)
    split_w = st.number_input("Home / away %", 0, 100, 20)
    roster_w = st.number_input("Roster / injuries %", 0, 100, 20)

    st.markdown("### Home / away split weights")
    win_w = st.number_input("Win % share", 0, 100, 25)
    ats_w = st.number_input("ATS % share", 0, 100, 30)
    margin_w = st.number_input("Scoring margin share", 0, 100, 25)
    sos_w = st.number_input("Schedule strength share", 0, 100, 20)

if recent_w + matchup_w + split_w + roster_w != 100:
    st.warning("Main model weights must add up to 100%.")
if win_w + ats_w + margin_w + sos_w != 100:
    st.warning("Home/away split weights must add up to 100%.")

st.subheader(f"{league} — {day:%B %d, %Y}")

try:
    games = get_games(league, day)
    st.success(f"Schedule connected: {len(games)} games found")
except Exception as error:
    games = []
    st.error(f"Schedule connection failed: {error}")

matchups = []
for game in games:
    competitions = game.get("competitions", [])
    if not competitions:
        continue
    comp = competitions[0]
    teams = comp.get("competitors", [])
    home = next((t for t in teams if t.get("homeAway") == "home"), None)
    away = next((t for t in teams if t.get("homeAway") == "away"), None)
    if not home or not away:
        continue

    home_name = home["team"]["displayName"]
    away_name = away["team"]["displayName"]
    status = game.get("status", {}).get("type", {}).get("description", "")
    matchups.append({
        "label": f"{away_name} @ {home_name}",
        "home": home_name,
        "away": away_name,
        "status": status,
        "date": game.get("date", ""),
        "home_score": home.get("score", ""),
        "away_score": away.get("score", "")
    })

if matchups:
    st.dataframe(
        [{
            "Game": m["label"],
            "Status": m["status"],
            "Away score": m["away_score"],
            "Home score": m["home_score"]
        } for m in matchups],
        use_container_width=True,
        hide_index=True
    )
    selected = st.selectbox(
        "Choose a matchup",
        range(len(matchups)),
        format_func=lambda i: matchups[i]["label"]
    )
    game = matchups[selected]
    home_name, away_name = game["home"], game["away"]
else:
    st.info("You can still evaluate a game by entering teams manually.")
    home_name = st.text_input("Home team", "Home")
    away_name = st.text_input("Away team", "Away")

st.divider()
st.subheader(f"Model: {away_name} at {home_name}")
st.caption("Positive adjustments favor home; negative adjustments favor away.")

c1, c2 = st.columns(2)
with c1:
    home_strength = st.number_input("Home strength (0–100)", 0., 100., 50., step=1.)
    recent = st.number_input("Recent-form advantage (points)", value=0., step=0.5)
    split = st.number_input("Home/away advantage (points)", value=0., step=0.5)
    injuries = st.number_input("Injury impact (points)", value=0., step=0.5)
    market_spread = st.number_input(
        "Home sportsbook spread (e.g. -3.5)", value=0., step=0.5
    )
with c2:
    away_strength = st.number_input("Away strength (0–100)", 0., 100., 50., step=1.)
    matchup = st.number_input("Matchup advantage (points)", value=0., step=0.5)
    roster = st.number_input("Roster advantage (points)", value=0., step=0.5)
    rest = st.number_input("Rest/travel impact (points)", value=0., step=0.5)
    home_ml = st.number_input("Home moneyline (American)", value=-110, step=5)

market_total = st.number_input("Sportsbook total", min_value=0., value=0., step=0.5)
projected_total = st.number_input(
    "Independent projected total (0 = unavailable)",
    min_value=0., value=0., step=0.5
)

base = HOME_EDGE[league]
strength_scale = 0.15 if league not in ("NHL", "MLB", "SOCCER") else 0.025

projected_margin = (
    base
    + (home_strength - away_strength) * strength_scale
    + recent * recent_w / 100
    + matchup * matchup_w / 100
    + split * split_w / 100
    + roster * roster_w / 100
    + injuries
    + rest
)

sd = MARGIN_SD[league]
home_win = normal_cdf(projected_margin / sd)
home_cover = normal_cdf((projected_margin + market_spread) / sd)
home_ev = value_ev(home_win, home_ml)

st.divider()
a, b, c = st.columns(3)
a.metric("Projected home margin", f"{projected_margin:+.1f}")
b.metric("Home win estimate", f"{home_win:.1%}")
c.metric("Home spread cover estimate", f"{home_cover:.1%}")

st.write(f"Estimated home ML EV: **{home_ev:+.1f}%**")
st.write(f"Estimated away spread cover: **{1-home_cover:.1%}**")

if projected_total > 0 and market_total > 0:
    st.write(f"Total edge: **{projected_total - market_total:+.1f} points**")
else:
    st.info("Enter an independent total projection to calculate a totals edge.")

st.warning(
    "Exploratory model only. Strength ratings and adjustment inputs "
    "are not automatically sourced. Probabilities are uncalibrated, "
    "and sportsbook odds must be entered manually."
)
