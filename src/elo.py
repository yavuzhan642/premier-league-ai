from statistics import median

from data_loader import (
    load_seasons,
    load_current_season,
    load_championship_season,
)


STARTING_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 60

PROMOTED_TEAMS = {
    "Hull",
    "Coventry",
    "Ipswich",
}


def expected_score(rating_a, rating_b):
    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def update_elo(
    home_elo,
    away_elo,
    home_goals,
    away_goals,
):
    adjusted_home_elo = (
        home_elo + HOME_ADVANTAGE
    )

    expected_home = expected_score(
        adjusted_home_elo,
        away_elo,
    )

    expected_away = 1 - expected_home

    if home_goals > away_goals:
        actual_home = 1
        actual_away = 0

    elif home_goals < away_goals:
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5

    new_home_elo = home_elo + K_FACTOR * (
        actual_home - expected_home
    )

    new_away_elo = away_elo + K_FACTOR * (
        actual_away - expected_away
    )

    return new_home_elo, new_away_elo


def calculate_elo_history(
    matches,
    initial_ratings=None,
):
    matches = (
        matches
        .sort_values("date")
        .copy()
    )

    if initial_ratings is None:
        ratings = {}
    else:
        ratings = initial_ratings.copy()

    home_elos = []
    away_elos = []

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        if home_team not in ratings:
            ratings[home_team] = STARTING_ELO

        if away_team not in ratings:
            ratings[away_team] = STARTING_ELO

        home_elo = ratings[home_team]
        away_elo = ratings[away_team]

        # Maçtan önceki rating.
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        new_home_elo, new_away_elo = update_elo(
            home_elo,
            away_elo,
            int(match["fthg"]),
            int(match["ftag"]),
        )

        ratings[home_team] = new_home_elo
        ratings[away_team] = new_away_elo

    matches["home_elo"] = home_elos
    matches["away_elo"] = away_elos

    matches["elo_difference"] = (
        matches["home_elo"]
        - matches["away_elo"]
    )

    return matches, ratings


def build_championship_ratings(
    pl_ratings,
):
    championship = (
        load_championship_season(2025)
    )

    championship = championship[
        championship["status"] == "FINISHED"
    ].copy()

    championship_teams = (
        set(championship["home_team"])
        | set(championship["away_team"])
    )

    # Geçmiş Premier League verisinde bulunan
    # Championship takımlarını buluyoruz.
    bridge_ratings = {
        team: pl_ratings[team]
        for team in championship_teams
        if team in pl_ratings
    }

    print()
    print("PL geçmişi olan Championship takımları:")
    print("--------------------------------------")

    for team, rating in sorted(
        bridge_ratings.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{team:22} {rating:.1f}"
        )

    # Premier League geçmişi olmayan Championship
    # takımlarına, bildiğimiz Championship
    # takımlarının medyan ratingini veriyoruz.
    championship_base = median(
        bridge_ratings.values()
    )

    print()
    print(
        "Championship başlangıç tabanı:",
        round(championship_base, 1),
    )

    initial_ratings = {}

    for team in championship_teams:
        initial_ratings[team] = (
            bridge_ratings.get(
                team,
                championship_base,
            )
        )

    _, championship_ratings = (
        calculate_elo_history(
            championship,
            initial_ratings,
        )
    )

    return championship_ratings


def build_current_ratings():
    historical_seasons = [
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]

    historical_matches = load_seasons(
        historical_seasons
    )

    # 2025/26 sonundaki PL Elo değerleri.
    _, pl_ratings = calculate_elo_history(
        historical_matches
    )

    # 2025/26 Championship'i ayrıca oynat.
    championship_ratings = (
        build_championship_ratings(
            pl_ratings
        )
    )

    current_season = (
        load_current_season()
    )

    current_teams = (
        set(current_season["home_team"])
        | set(current_season["away_team"])
    )

    # 2026/27 başlangıç ratingleri.
    starting_ratings = {}

    for team in current_teams:

        if team in PROMOTED_TEAMS:
            starting_ratings[team] = (
                championship_ratings[team]
            )

        elif team in pl_ratings:
            starting_ratings[team] = (
                pl_ratings[team]
            )

        else:
            starting_ratings[team] = (
                STARTING_ELO
            )

    print()
    print("Yükselen takımlar - sezon başı Elo")
    print("--------------------------------")

    for team in sorted(PROMOTED_TEAMS):
        print(
            f"{team:15} "
            f"{starting_ratings[team]:.1f}"
        )

    # Sadece gerçekten oynanmış 2026/27
    # maçlarını Elo'ya işliyoruz.
    finished = current_season[
        current_season["status"] == "FINISHED"
    ].copy()

    _, current_ratings = (
        calculate_elo_history(
            finished,
            starting_ratings,
        )
    )

    return current_ratings


def main():
    ratings = build_current_ratings()

    print()
    print("Güncel 2026/27 Elo")
    print("------------------")

    sorted_ratings = sorted(
        ratings.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for position, (team, rating) in enumerate(
        sorted_ratings,
        start=1,
    ):
        print(
            f"{position:2}. "
            f"{team:20} "
            f"{rating:.1f}"
        )


if __name__ == "__main__":
    main()