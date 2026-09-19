import premier_league_data as pl
import pandas as pd


def load_all_matches():
    return pl.load_results()


def load_season(season_name):
    matches = load_all_matches()

    season = matches[
        matches["season"] == season_name
    ].copy()

    return season


def main():
    season = load_season("2025-26")

    print("2025/26 Premier League")
    print("----------------------")
    print("Maç sayısı:", len(season))
    print("Takım sayısı:", season["home_team"].nunique())

    print("\nTakımlar:")
    print(sorted(season["home_team"].unique()))

def load_seasons(season_names):
    matches = load_all_matches()

    selected = matches[
        matches["season"].isin(season_names)
    ].copy()

    selected["date"] = pd.to_datetime(selected["date"])

    return selected.sort_values("date")


if __name__ == "__main__":
    main()