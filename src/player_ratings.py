import re
import unicodedata
from pathlib import Path

import pandas as pd

from player_data import load_player_data


FC27_PATH = Path(
    "data/processed/fc27_premier_league_ratings.csv"
)


def normalize_name(name):
    """
    João Pedro -> joao pedro
    Martin Ødegaard -> martin odegaard

    Noktalama ve aksan farklarını
    mümkün olduğunca kaldırır.
    """

    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    replacements = {
        "ø": "o",
        "ð": "d",
        "þ": "th",
        "ł": "l",
        "đ": "d",
        "æ": "ae",
        "œ": "oe",
    }

    for old, new in replacements.items():
        name = name.replace(
            old,
            new,
        )

    name = unicodedata.normalize(
        "NFKD",
        name,
    )

    name = "".join(
        char
        for char in name
        if not unicodedata.combining(
            char
        )
    )

    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name.strip()


def load_fc27_ratings():
    if not FC27_PATH.exists():
        raise RuntimeError(
            "FC27 ratings CSV bulunamadı. "
            "Önce python src/fc27_ratings.py çalıştır."
        )

    ratings = pd.read_csv(
        FC27_PATH
    )

    ratings["name_key"] = (
        ratings["player_name"]
        .apply(normalize_name)
    )

    return ratings


def build_player_ratings():
    players = load_player_data().copy()
    ratings = load_fc27_ratings().copy()

    players["name_key"] = (
        players["player_name"]
        .apply(normalize_name)
    )

    # FC tarafında aynı isimden birden fazla
    # oyuncu varsa şimdilik en yüksek
    # overall olan kaydı tut.
    ratings = (
        ratings
        .sort_values(
            "overall",
            ascending=False,
        )
        .drop_duplicates(
            subset=["name_key"]
        )
    )

    rating_columns = [
        "name_key",
        "overall",
        "fc_position",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physical",
    ]

    merged = players.merge(
        ratings[rating_columns],
        on="name_key",
        how="left",
    )

    merged["has_fc27_rating"] = (
        merged["overall"]
        .notna()
    )

    return merged


def main():
    players = build_player_ratings()

    total = len(players)

    matched = int(
        players[
            "has_fc27_rating"
        ].sum()
    )

    unmatched = (
        total - matched
    )

    print()
    print("FC27 EŞLEŞTİRME")
    print("===============")

    print(
        "Football-Data oyuncusu:",
        total,
    )

    print(
        "FC27 ile eşleşen:",
        matched,
    )

    print(
        "Eşleşmeyen:",
        unmatched,
    )

    print(
        "Eşleşme oranı:",
        f"{matched / total:.1%}",
    )

    print()
    print("EN YÜKSEK OVERALL")
    print("-----------------")

    print(
        players[
            players["has_fc27_rating"]
        ]
        .sort_values(
            "overall",
            ascending=False,
        )[
            [
                "player_name",
                "team",
                "position",
                "fc_position",
                "overall",
                "pace",
                "shooting",
                "passing",
                "dribbling",
                "defending",
                "physical",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    print()
    print("FC27'DE EŞLEŞMEYEN İLK 50")
    print("-------------------------")

    print(
        players[
            ~players["has_fc27_rating"]
        ][
            [
                "player_name",
                "team",
                "position",
            ]
        ]
        .head(50)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()