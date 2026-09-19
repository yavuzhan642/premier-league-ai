from pathlib import Path
import sys
import numpy as np
import pandas as pd

from data_loader import (
    load_seasons,
    load_current_season,
    load_championship_season,
)
from elo import (
    build_current_ratings,
    update_elo,
)
from goal_model import (
    FEATURES,
    create_poisson_model,
)
from standings import create_standings


DATA_PATH = Path(
    "data/processed/match_features.csv"
)


HISTORICAL_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


def normalize_dates(matches):
    matches = matches.copy()

    matches["date"] = (
        pd.to_datetime(
            matches["date"],
            utc=True,
        )
        .dt.tz_convert(None)
    )

    return matches


def train_goal_models():
    dataset = pd.read_csv(
        DATA_PATH
    )

    train = dataset[
        (dataset["home_last5_games"] == 5)
        & (dataset["away_last5_games"] == 5)
    ].copy()

    X = train[FEATURES]

    home_model = create_poisson_model()
    away_model = create_poisson_model()

    home_model.fit(
        X,
        train["home_goals"],
    )

    away_model.fit(
        X,
        train["away_goals"],
    )

    return home_model, away_model


def add_match_to_form(
    form_state,
    home_team,
    away_team,
    home_goals,
    away_goals,
):
    if home_goals > away_goals:
        home_points = 3
        away_points = 0

    elif home_goals < away_goals:
        home_points = 0
        away_points = 3

    else:
        home_points = 1
        away_points = 1

    form_state[home_team].append(
        {
            "points": home_points,
            "goals_for": home_goals,
            "goals_against": away_goals,
        }
    )

    form_state[away_team].append(
        {
            "points": away_points,
            "goals_for": away_goals,
            "goals_against": home_goals,
        }
    )

    # Sadece son 5 maçı tut.
    form_state[home_team] = (
        form_state[home_team][-5:]
    )

    form_state[away_team] = (
        form_state[away_team][-5:]
    )


def build_form_state(
    current_teams,
    historical_matches,
):
    form_state = {
        team: []
        for team in current_teams
    }

    historical_matches = (
        historical_matches
        .sort_values("date")
    )

    for _, match in historical_matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        # Sadece güncel Premier League
        # takımlarıyla ilgileniyoruz.
        if (
            home not in form_state
            and away not in form_state
        ):
            continue

        if pd.isna(match["fthg"]):
            continue

        home_goals = int(
            match["fthg"]
        )

        away_goals = int(
            match["ftag"]
        )

        if home in form_state:
            if home_goals > away_goals:
                points = 3
            elif home_goals == away_goals:
                points = 1
            else:
                points = 0

            form_state[home].append(
                {
                    "points": points,
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                }
            )

            form_state[home] = (
                form_state[home][-5:]
            )

        if away in form_state:
            if away_goals > home_goals:
                points = 3
            elif away_goals == home_goals:
                points = 1
            else:
                points = 0

            form_state[away].append(
                {
                    "points": points,
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                }
            )

            form_state[away] = (
                form_state[away][-5:]
            )

    return form_state


def get_form_features(form):
    return {
        "points": sum(
            game["points"]
            for game in form
        ),
        "goals_for": sum(
            game["goals_for"]
            for game in form
        ),
        "goals_against": sum(
            game["goals_against"]
            for game in form
        ),
    }


def predict_expected_goals(
    home_model,
    away_model,
    home_team,
    away_team,
    ratings,
    form_state,
):
    home_form = get_form_features(
        form_state[home_team]
    )

    away_form = get_form_features(
        form_state[away_team]
    )

    feature_row = {
        "home_elo": ratings[home_team],
        "away_elo": ratings[away_team],

        "elo_difference": (
            ratings[home_team]
            - ratings[away_team]
        ),

        "home_last5_points": (
            home_form["points"]
        ),

        "home_last5_goals_for": (
            home_form["goals_for"]
        ),

        "home_last5_goals_against": (
            home_form["goals_against"]
        ),

        "away_last5_points": (
            away_form["points"]
        ),

        "away_last5_goals_for": (
            away_form["goals_for"]
        ),

        "away_last5_goals_against": (
            away_form["goals_against"]
        ),
    }

    X = pd.DataFrame(
        [feature_row]
    )[FEATURES]

    home_expected = float(
        home_model.predict(X)[0]
    )

    away_expected = float(
        away_model.predict(X)[0]
    )

    # Modelin aşırı uç değer üretmesine
    # karşı küçük bir güvenlik sınırı.
    home_expected = float(
        np.clip(
            home_expected,
            0.05,
            4.5,
        )
    )

    away_expected = float(
        np.clip(
            away_expected,
            0.05,
            4.5,
        )
    )

    return (
        home_expected,
        away_expected,
    )


def simulate_season(seed=None):
    # Seed verilmezse her çalıştırmada
    # farklı bir sezon üret.
    if seed is None:
        seed = int(
            np.random
            .SeedSequence()
            .generate_state(1)[0]
        )

    rng = np.random.default_rng(
        seed
    )

    print(
        "Goal modelleri eğitiliyor..."
    )

    home_model, away_model = (
        train_goal_models()
    )

    print(
        "Güncel sezon yükleniyor..."
    )

    current_season = normalize_dates(
        load_current_season()
    )

    finished = current_season[
        current_season["status"]
        == "FINISHED"
    ].copy()

    upcoming = current_season[
        current_season["status"]
        != "FINISHED"
    ].copy()

    current_teams = sorted(
        set(current_season["home_team"])
        | set(current_season["away_team"])
    )

    print(
        "Güncel Elo hesaplanıyor..."
    )

    ratings = (
        build_current_ratings()
    )

    starting_ratings = ratings.copy()

    # ----------------------------------
    # SON 5 FORM İÇİN GEÇMİŞ MAÇLAR
    # ----------------------------------

    historical_pl = normalize_dates(
        load_seasons(
            HISTORICAL_SEASONS
        )
    )

    championship = normalize_dates(
        load_championship_season(2025)
    )

    championship = championship[
        championship["status"]
        == "FINISHED"
    ]

    known_matches = pd.concat(
        [
            historical_pl,
            championship,
            finished,
        ],
        ignore_index=True,
        sort=False,
    )

    form_state = build_form_state(
        current_teams,
        known_matches,
    )

    # ----------------------------------
    # KALAN MAÇLARI OYNA
    # ----------------------------------

    simulated_matches = []

    upcoming = upcoming.sort_values(
        "date"
    )

    for _, match in upcoming.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        (
            home_expected,
            away_expected,
        ) = predict_expected_goals(
            home_model,
            away_model,
            home_team,
            away_team,
            ratings,
            form_state,
        )

        home_goals = int(
            rng.poisson(
                home_expected
            )
        )

        away_goals = int(
            rng.poisson(
                away_expected
            )
        )

        if home_goals > away_goals:
            result = "H"

        elif home_goals < away_goals:
            result = "A"

        else:
            result = "D"

        simulated_matches.append(
            {
                "date": match["date"],
                "home_team": home_team,
                "away_team": away_team,
                "fthg": home_goals,
                "ftag": away_goals,
                "ftr": result,
                "home_expected_goals": (
                    home_expected
                ),
                "away_expected_goals": (
                    away_expected
                ),
            }
        )

        # ------------------------------
        # BU EVRENİN FORMUNU GÜNCELLE
        # ------------------------------

        add_match_to_form(
            form_state,
            home_team,
            away_team,
            home_goals,
            away_goals,
        )

        # ------------------------------
        # BU EVRENİN ELO'SUNU GÜNCELLE
        # ------------------------------

        (
            new_home_elo,
            new_away_elo,
        ) = update_elo(
            ratings[home_team],
            ratings[away_team],
            home_goals,
            away_goals,
        )

        ratings[home_team] = (
            new_home_elo
        )

        ratings[away_team] = (
            new_away_elo
        )

    simulated = pd.DataFrame(
        simulated_matches
    )

    # ----------------------------------
    # GERÇEK + SİMÜLE EDİLEN MAÇLAR
    # ----------------------------------

    actual_matches = finished[
        [
            "date",
            "home_team",
            "away_team",
            "fthg",
            "ftag",
            "ftr",
        ]
    ].copy()

    simulated_for_table = simulated[
        [
            "date",
            "home_team",
            "away_team",
            "fthg",
            "ftag",
            "ftr",
        ]
    ].copy()

    full_season = pd.concat(
        [
            actual_matches,
            simulated_for_table,
        ],
        ignore_index=True,
    )

    table = create_standings(
        full_season
    )

    return {
        "seed": seed,
        "table": table,
        "simulated_matches": simulated,
        "starting_ratings": starting_ratings,
        "final_ratings": ratings,
    }

def main():
    seed = None

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])

    result = simulate_season(
        seed=seed
    )

    seed = result["seed"]
    table = result["table"]

    print()
    print("=" * 55)
    print("ALTERNATİF 2026/27 PREMIER LEAGUE")
    print("=" * 55)

    print()
    print("Scenario seed:", seed)

    print()
    print(
        table[
            [
                "team",
                "played",
                "won",
                "drawn",
                "lost",
                "goal_difference",
                "points",
            ]
        ].to_string()
    )

    champion = table.iloc[0]["team"]

    relegated = table.tail(3)[
        "team"
    ].tolist()

    print()
    print(
        "🏆 Şampiyon:",
        champion,
    )

    print(
        "⬇️ Küme düşenler:",
        ", ".join(relegated),
    )

    starting_ratings = result["starting_ratings"]
    final_ratings = result["final_ratings"]

    # Sezon başındaki güç sırası
    starting_order = sorted(
        starting_ratings,
        key=starting_ratings.get,
        reverse=True,
    )

    starting_rank = {
        team: rank
        for rank, team in enumerate(
            starting_order,
            start=1,
        )
    }

    # Final lig sırası
    final_rank = {
        row["team"]: rank
        for rank, (_, row) in enumerate(
            table.iterrows(),
            start=1,
        )
    }

    rank_changes = {}

    for team in final_rank:
        rank_changes[team] = (
            starting_rank[team]
            - final_rank[team]
        )

    dark_horse = max(
        rank_changes,
        key=rank_changes.get,
    )

    disappointment = min(
        rank_changes,
        key=rank_changes.get,
    )

    elo_changes = {
        team: (
            final_ratings[team]
            - starting_ratings[team]
        )
        for team in final_ratings
    }

    biggest_elo_riser = max(
        elo_changes,
        key=elo_changes.get,
    )

    biggest_elo_faller = min(
        elo_changes,
        key=elo_changes.get,
    )

    print()
    print("SEZON HİKÂYESİ")
    print("--------------")

    print(
        "🌟 Dark horse:",
        dark_horse,
        f"({starting_rank[dark_horse]}. güç → "
        f"{final_rank[dark_horse]}. sıra)",
    )

    print(
        "📉 En büyük düşüş:",
        disappointment,
        f"({starting_rank[disappointment]}. güç → "
        f"{final_rank[disappointment]}. sıra)",
    )

    print(
        "📈 En çok Elo kazanan:",
        biggest_elo_riser,
        f"{elo_changes[biggest_elo_riser]:+.1f}",
    )

    print(
        "📉 En çok Elo kaybeden:",
        biggest_elo_faller,
        f"{elo_changes[biggest_elo_faller]:+.1f}",
    )

if __name__ == "__main__":
    main()