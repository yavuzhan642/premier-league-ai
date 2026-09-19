import pandas as pd

from data_loader import load_season


def create_standings(matches):
    teams = sorted(
        set(matches["home_team"]) | set(matches["away_team"])
    )

    table = {}

    for team in teams:
        table[team] = {
            "team": team,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        home_goals = int(match["fthg"])
        away_goals = int(match["ftag"])

        table[home]["played"] += 1
        table[away]["played"] += 1

        table[home]["goals_for"] += home_goals
        table[home]["goals_against"] += away_goals

        table[away]["goals_for"] += away_goals
        table[away]["goals_against"] += home_goals

        if home_goals > away_goals:
            table[home]["won"] += 1
            table[away]["lost"] += 1
            table[home]["points"] += 3

        elif home_goals < away_goals:
            table[away]["won"] += 1
            table[home]["lost"] += 1
            table[away]["points"] += 3

        else:
            table[home]["drawn"] += 1
            table[away]["drawn"] += 1
            table[home]["points"] += 1
            table[away]["points"] += 1

    for team in teams:
        table[team]["goal_difference"] = (
            table[team]["goals_for"]
            - table[team]["goals_against"]
        )

    standings = pd.DataFrame(table.values())

    standings = standings.sort_values(
        by=["points", "goal_difference", "goals_for"],
        ascending=False,
    ).reset_index(drop=True)

    standings.index += 1

    return standings


def main():
    matches = load_season("2025-26")
    standings = create_standings(matches)

    print(standings)


if __name__ == "__main__":
    main()