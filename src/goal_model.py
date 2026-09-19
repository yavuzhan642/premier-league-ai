from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(
    "data/processed/match_features.csv"
)


FEATURES = [
    "home_elo",
    "away_elo",
    "elo_difference",

    "home_last5_points",
    "home_last5_goals_for",
    "home_last5_goals_against",

    "away_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
]


def create_poisson_model():
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "poisson",
                PoissonRegressor(
                    alpha=0.1,
                    max_iter=1000,
                ),
            ),
        ]
    )

    return model

def simulate_score(
    home_expected_goals,
    away_expected_goals,
):
    home_goals = np.random.poisson(
        home_expected_goals
    )

    away_goals = np.random.poisson(
        away_expected_goals
    )

    return int(home_goals), int(away_goals)

def main():
    dataset = pd.read_csv(
        DATA_PATH
    )

    train = dataset[
        dataset["season"] != "2025-26"
    ].copy()

    test = dataset[
        dataset["season"] == "2025-26"
    ].copy()

    # Yeterli son-5 verisi olmayan
    # maçları şimdilik çıkartıyoruz.
    train = train[
        (train["home_last5_games"] == 5)
        & (train["away_last5_games"] == 5)
    ]

    test = test[
        (test["home_last5_games"] == 5)
        & (test["away_last5_games"] == 5)
    ]

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    # İki ayrı model:
    # ev sahibi kaç gol atar?
    # deplasman kaç gol atar?

    home_model = create_poisson_model()
    away_model = create_poisson_model()

    home_model.fit(
        X_train,
        train["home_goals"],
    )

    away_model.fit(
        X_train,
        train["away_goals"],
    )

    predicted_home_goals = (
        home_model.predict(X_test)
    )

    predicted_away_goals = (
        away_model.predict(X_test)
    )

    home_mae = mean_absolute_error(
        test["home_goals"],
        predicted_home_goals,
    )

    away_mae = mean_absolute_error(
        test["away_goals"],
        predicted_away_goals,
    )

    print("Premier League Goal Model")
    print("-------------------------")

    print(
        "Train maç:",
        len(train),
    )

    print(
        "Test maç:",
        len(test),
    )

    print()

    print(
        "Home goal MAE:",
        round(home_mae, 3),
    )

    print(
        "Away goal MAE:",
        round(away_mae, 3),
    )

    results = test[
        [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    results["home_xg_model"] = (
        predicted_home_goals
    )

    results["away_xg_model"] = (
        predicted_away_goals
    )

    print()
    print("İlk 20 maç:")
    print(
        results.head(20).to_string(
            index=False
        )
    )

    print()
    print("Aynı maç için 10 farklı simülasyon")
    print("--------------------------------")

    example = results.iloc[0]

    home_team = example["home_team"]
    away_team = example["away_team"]

    home_expected = example["home_xg_model"]
    away_expected = example["away_xg_model"]

    print(
        f"{home_team} vs {away_team}"
    )

    print(
        "Beklenen goller:",
        round(home_expected, 2),
        "-",
        round(away_expected, 2),
    )

    print()

    for simulation_number in range(1, 11):
        home_goals, away_goals = simulate_score(
            home_expected,
            away_expected,
        )

        print(
            f"Run {simulation_number:2}: "
            f"{home_team} {home_goals} - "
            f"{away_goals} {away_team}"
        )


if __name__ == "__main__":
    main()