from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


DATA_PATH = Path("data/processed/match_features.csv")


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


def load_dataset():
    return pd.read_csv(DATA_PATH)


def main():
    dataset = load_dataset()

    # Eğitim: 2021/22 - 2024/25
    train = dataset[
        dataset["season"] != "2025-26"
    ].copy()

    # Test: modelin hiç görmeyeceği 2025/26
    test = dataset[
        dataset["season"] == "2025-26"
    ].copy()

    # Takımların yeterli geçmiş formu olmayan maçları
    # şimdilik çıkartıyoruz.
    train = train[
        (train["home_last5_games"] == 5)
        & (train["away_last5_games"] == 5)
    ]

    test = test[
        (test["home_last5_games"] == 5)
        & (test["away_last5_games"] == 5)
    ]

    X_train = train[FEATURES]
    y_train = train["result"]

    X_test = test[FEATURES]
    y_test = test["result"]

    model = LogisticRegression(
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=model.classes_,
    )

    print("Premier League Match Predictor")
    print("------------------------------")

    print("Train maç sayısı:", len(train))
    print("Test maç sayısı:", len(test))

    print()
    print("Accuracy:", round(accuracy, 3))
    print("Log Loss:", round(loss, 3))

    print()
    print("Model sınıfları:")
    print(model.classes_)

    results = test[
        [
            "date",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    results["prediction"] = predictions

    results["prob_away"] = probabilities[
        :,
        list(model.classes_).index("A")
    ]

    results["prob_draw"] = probabilities[
        :,
        list(model.classes_).index("D")
    ]

    results["prob_home"] = probabilities[
        :,
        list(model.classes_).index("H")
    ]

    print()
    print("İlk 20 tahmin:")
    print(
        results.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()