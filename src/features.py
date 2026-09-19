import pandas as pd
from pathlib import Path

from data_loader import load_seasons


def get_team_recent_form(matches, team, before_date, last_n=5):
    previous_matches = matches[
        (
            (matches["home_team"] == team)
            | (matches["away_team"] == team)
        )
        & (matches["date"] < before_date)
    ].sort_values("date")

    previous_matches = previous_matches.tail(last_n)

    points = 0
    goals_for = 0
    goals_against = 0

    for _, match in previous_matches.iterrows():
        if match["home_team"] == team:
            team_goals = int(match["fthg"])
            opponent_goals = int(match["ftag"])
        else:
            team_goals = int(match["ftag"])
            opponent_goals = int(match["fthg"])

        goals_for += team_goals
        goals_against += opponent_goals

        if team_goals > opponent_goals:
            points += 3
        elif team_goals == opponent_goals:
            points += 1

    return {
        "games": len(previous_matches),
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def build_feature_dataset(matches):
    matches = matches.sort_values("date").copy()

    rows = []

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["date"]

        home_form = get_team_recent_form(
            matches,
            home_team,
            match_date,
        )

        away_form = get_team_recent_form(
            matches,
            away_team,
            match_date,
        )

        row = {
            "season": match["season"],
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,

            "home_last5_games": home_form["games"],
            "home_last5_points": home_form["points"],
            "home_last5_goals_for": home_form["goals_for"],
            "home_last5_goals_against": home_form["goals_against"],

            "away_last5_games": away_form["games"],
            "away_last5_points": away_form["points"],
            "away_last5_goals_for": away_form["goals_for"],
            "away_last5_goals_against": away_form["goals_against"],

            "home_goals": int(match["fthg"]),
            "away_goals": int(match["ftag"]),
            "result": match["ftr"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    seasons = [
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]

    matches = load_seasons(seasons)

    dataset = build_feature_dataset(matches)

    print(dataset.head(10).to_string())

    print()
    print("Toplam maç:", len(dataset))

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    dataset.to_csv(
        "data/processed/match_features.csv",
        index=False,
    )

    print("Dataset kaydedildi:")
    print("data/processed/match_features.csv")


if __name__ == "__main__":
    main()