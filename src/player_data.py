import pandas as pd
import requests

from data_loader import (
    API_KEY,
    BASE_URL,
    normalize_team_name,
)


SEASON_START_YEAR = 2026


def get_headers():
    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY bulunamadı."
        )

    return {
        "X-Auth-Token": API_KEY,
    }


def load_squads():
    url = (
        f"{BASE_URL}/competitions/PL/teams"
    )

    response = requests.get(
        url,
        headers=get_headers(),
        params={
            "season": SEASON_START_YEAR,
        },
        timeout=30,
    )

    print(
        "Squads API status:",
        response.status_code,
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for team in data.get("teams", []):
        team_name = normalize_team_name(
            team["name"]
        )

        for player in team.get("squad", []):
            rows.append(
                {
                    "player_id": player["id"],
                    "player_name": player["name"],
                    "team": team_name,
                    "position": player.get(
                        "position"
                    ),
                    "date_of_birth": player.get(
                        "dateOfBirth"
                    ),
                    "nationality": player.get(
                        "nationality"
                    ),
                }
            )

    return pd.DataFrame(rows)


def load_scorers():
    url = (
        f"{BASE_URL}/competitions/PL/scorers"
    )

    response = requests.get(
        url,
        headers=get_headers(),
        params={
            "season": SEASON_START_YEAR,
            "limit": 200,
        },
        timeout=30,
    )

    print(
        "Scorers API status:",
        response.status_code,
    )

    response.raise_for_status()

    data = response.json()

    rows = []

    for scorer in data.get(
        "scorers",
        []
    ):
        player = scorer["player"]
        team = scorer["team"]

        rows.append(
            {
                "player_id": player["id"],
                "scorer_team": normalize_team_name(
                    team["name"]
                ),
                "real_goals": (
                    scorer.get("goals")
                    or 0
                ),
                "real_assists": (
                    scorer.get("assists")
                    or 0
                ),
                "real_penalties": (
                    scorer.get("penalties")
                    or 0
                ),
                "played_matches": (
                    scorer.get(
                        "playedMatches"
                    )
                    or 0
                ),
            }
        )

    return pd.DataFrame(rows)


def load_player_data():
    squads = load_squads()
    scorers = load_scorers()

    players = squads.merge(
        scorers,
        on="player_id",
        how="left",
    )

    numeric_columns = [
        "real_goals",
        "real_assists",
        "real_penalties",
        "played_matches",
    ]

    for column in numeric_columns:
        players[column] = (
            players[column]
            .fillna(0)
            .astype(int)
        )

    # Merge sonrası artık buna
    # ihtiyacımız yok.
    if "scorer_team" in players.columns:
        players = players.drop(
            columns=["scorer_team"]
        )

    return players


def main():
    players = load_player_data()

    print()
    print("OYUNCU VERİSİ")
    print("-------------")

    print(
        "Toplam oyuncu:",
        len(players),
    )

    print(
        "Takım sayısı:",
        players["team"].nunique(),
    )

    print()

    print("Pozisyonlar:")
    print(
        players["position"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Mevcut gol krallığı:")
    print(
        players.sort_values(
            [
                "real_goals",
                "real_assists",
            ],
            ascending=False,
        )[
            [
                "player_name",
                "team",
                "position",
                "real_goals",
                "real_assists",
                "played_matches",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()