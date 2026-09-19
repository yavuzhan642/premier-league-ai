import os

import pandas as pd
import premier_league_data as pl
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

BASE_URL = "https://api.football-data.org/v4"


TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)


def result_from_score(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


# --------------------------------------------------
# GEÇMİŞ SEZONLAR
# --------------------------------------------------

def load_historical_matches():
    matches = pl.load_results().copy()

    matches["date"] = pd.to_datetime(
        matches["date"]
    )

    matches["status"] = "FINISHED"
    matches["competition"] = "PL"

    return matches


def load_historical_season(season_name):
    matches = load_historical_matches()

    season = matches[
        matches["season"] == season_name
    ].copy()

    return season.sort_values("date")


# --------------------------------------------------
# 2026/27 GÜNCEL SEZON
# --------------------------------------------------

def load_current_season():
    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY .env dosyasında bulunamadı."
        )

    url = (
        f"{BASE_URL}/competitions/PL/matches"
    )

    headers = {
        "X-Auth-Token": API_KEY,
    }

    params = {
        "season": 2026,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    print(
        "API status:",
        response.status_code,
    )

    print(
        "Kalan istek:",
        response.headers.get(
            "X-Requests-Available-Minute"
        ),
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for match in data["matches"]:
        full_time = match["score"]["fullTime"]

        home_goals = full_time["home"]
        away_goals = full_time["away"]

        rows.append(
            {
                "match_id": match["id"],
                "competition": "PL",
                "season": "2026-27",
                "date": pd.to_datetime(
                    match["utcDate"]
                ),
                "status": match["status"],
                "matchday": match.get(
                    "matchday"
                ),
                "home_team": normalize_team_name(
                    match["homeTeam"]["name"]
                ),
                "away_team": normalize_team_name(
                    match["awayTeam"]["name"]
                ),
                "fthg": home_goals,
                "ftag": away_goals,
                "ftr": result_from_score(
                    home_goals,
                    away_goals,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .reset_index(drop=True)
    )

def load_championship_season(
    start_year=2025,
):
    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY .env dosyasında bulunamadı."
        )

    url = (
        f"{BASE_URL}/competitions/ELC/matches"
    )

    headers = {
        "X-Auth-Token": API_KEY,
    }

    params = {
        "season": start_year,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    print(
        "Championship API status:",
        response.status_code,
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for match in data["matches"]:
        full_time = match["score"]["fullTime"]

        home_goals = full_time["home"]
        away_goals = full_time["away"]

        rows.append(
            {
                "match_id": match["id"],
                "competition": "ELC",
                "season": f"{start_year}-{str(start_year + 1)[-2:]}",
                "date": pd.to_datetime(
                    match["utcDate"]
                ),
                "status": match["status"],
                "matchday": match.get("matchday"),

                "home_team": normalize_team_name(
                    match["homeTeam"]["name"]
                ),

                "away_team": normalize_team_name(
                    match["awayTeam"]["name"]
                ),

                "fthg": home_goals,
                "ftag": away_goals,

                "ftr": result_from_score(
                    home_goals,
                    away_goals,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .reset_index(drop=True)
    )


# --------------------------------------------------
# ORTAK ARAYÜZ
# --------------------------------------------------

def load_season(season_name):
    if season_name == "2026-27":
        return load_current_season()

    return load_historical_season(
        season_name
    )


def load_seasons(season_names):
    seasons = []

    for season_name in season_names:
        season = load_season(
            season_name
        )

        seasons.append(season)

    return (
        pd.concat(
            seasons,
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def main():
    historical = load_season(
        "2025-26"
    )

    current = load_season(
        "2026-27"
    )

    finished = current[
        current["status"] == "FINISHED"
    ]

    upcoming = current[
        current["status"] != "FINISHED"
    ]

    print()
    print("2025/26:")
    print(
        "Geçmiş maç:",
        len(historical),
    )

    print()
    print("2026/27:")
    print(
        "Toplam fikstür:",
        len(current),
    )
    print(
        "Oynanmış:",
        len(finished),
    )
    print(
        "Kalan:",
        len(upcoming),
    )


if __name__ == "__main__":
    main()