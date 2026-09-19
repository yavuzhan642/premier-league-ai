import sys

import numpy as np
import pandas as pd

from player_data import load_player_data
from simulator import simulate_season


ASSIST_PROBABILITY = 0.72


# Pozisyonun doğal gol üretme seviyesi.
GOAL_POSITION_BASE = {
    "Offence": 2.7,
    "Midfield": 0.85,
    "Defence": 0.16,
    "Goalkeeper": 0.002,
}


# Pozisyonun doğal asist üretme seviyesi.
ASSIST_POSITION_BASE = {
    "Offence": 2.2,
    "Midfield": 3.0,
    "Defence": 1.1,
    "Goalkeeper": 0.02,
}


# Basit 4-3-3 başlangıç dizilişi.
STARTER_TARGETS = {
    "Goalkeeper": 1,
    "Defence": 4,
    "Midfield": 3,
    "Offence": 3,
}


BENCH_SIZE = 5


def get_selection_weights(players):
    """
    İlk 11 / kadro seçiminde kullanılır.

    Mevcut sezonda gol/asist üretmiş oyuncuların
    oynama ihtimali biraz daha yüksek olur.

    simulated_minutes sayesinde sezon ilerledikçe
    düzenli oynayan oyuncuların rolü biraz oturur.
    """

    weights = (
        1.0
        + players["real_goals"].astype(float) * 1.0
        + players["real_assists"].astype(float) * 0.6
        + 0.10
        * np.sqrt(
            players["simulated_minutes"].astype(float)
            / 90.0
        )
    )

    return weights


def weighted_sample(
    candidates,
    count,
    rng,
):
    """
    Bir DataFrame içinden ağırlıklı ve
    tekrarsız oyuncu seçer.
    """

    if candidates.empty or count <= 0:
        return []

    count = min(
        count,
        len(candidates),
    )

    weights = get_selection_weights(
        candidates
    ).to_numpy(dtype=float)

    probabilities = (
        weights / weights.sum()
    )

    chosen = rng.choice(
        candidates.index.to_numpy(),
        size=count,
        replace=False,
        p=probabilities,
    )

    return chosen.tolist()


def select_match_players(
    players,
    team,
    rng,
):
    """
    Bir takım için:

    - 11 starter seçer
    - 5 kişilik yedek havuzu oluşturur
    - 3-5 oyuncuyu oyuna sokar
    - oyunculara dakika dağıtır

    Dönüş:
    minutes_by_player -> {index: dakika}
    starters -> starter index listesi
    """

    team_players = players[
        players["team"] == team
    ].copy()

    starters = []

    # -----------------------------
    # İLK 11
    # -----------------------------

    for position, target in (
        STARTER_TARGETS.items()
    ):
        candidates = team_players[
            (
                team_players["position"]
                == position
            )
            & (
                ~team_players.index.isin(
                    starters
                )
            )
        ]

        chosen = weighted_sample(
            candidates,
            target,
            rng,
        )

        starters.extend(chosen)

    # Pozisyon eksikliği varsa 11'e tamamla.
    if len(starters) < 11:

        remaining = team_players[
            ~team_players.index.isin(
                starters
            )
        ]

        missing = 11 - len(starters)

        extra = weighted_sample(
            remaining,
            missing,
            rng,
        )

        starters.extend(extra)

    # -----------------------------
    # YEDEKLER
    # -----------------------------

    remaining = team_players[
        ~team_players.index.isin(
            starters
        )
    ]

    # Şimdilik yedek kaleci sistemine girmiyoruz.
    # Gol/asist simülasyonu için 5 outfield bench.
    outfield_remaining = remaining[
        remaining["position"]
        != "Goalkeeper"
    ]

    bench = weighted_sample(
        outfield_remaining,
        BENCH_SIZE,
        rng,
    )

    # -----------------------------
    # DAKİKALAR
    # -----------------------------

    minutes = {
        player_index: 90
        for player_index in starters
    }

    # 3 ila 5 değişiklik.
    number_of_subs = int(
        rng.integers(
            3,
            min(5, len(bench)) + 1,
        )
    )

    if number_of_subs > 0:

        used_subs = rng.choice(
            bench,
            size=number_of_subs,
            replace=False,
        ).tolist()

    else:
        used_subs = []

    replaced_starters = set()

    for sub_index in used_subs:

        sub_position = players.at[
            sub_index,
            "position",
        ]

        # Önce aynı pozisyondan çıkan birini ara.
        same_position = [
            index
            for index in starters
            if (
                index
                not in replaced_starters
                and players.at[
                    index,
                    "position",
                ]
                == sub_position
                and players.at[
                    index,
                    "position",
                ]
                != "Goalkeeper"
            )
        ]

        if same_position:
            starter_index = rng.choice(
                same_position
            )

        else:
            # Olmazsa herhangi bir outfield starter.
            available = [
                index
                for index in starters
                if (
                    index
                    not in replaced_starters
                    and players.at[
                        index,
                        "position",
                    ]
                    != "Goalkeeper"
                )
            ]

            if not available:
                continue

            starter_index = rng.choice(
                available
            )

        replaced_starters.add(
            starter_index
        )

        # Oyuncu 55-85. dakikalar arasında çıkar.
        substitution_minute = int(
            rng.integers(
                55,
                86,
            )
        )

        minutes[starter_index] = (
            substitution_minute
        )

        minutes[sub_index] = (
            90 - substitution_minute
        )

    return minutes, starters


def get_goal_weights(
    match_players,
):
    """
    Golcü ağırlığı.

    Önemli değişiklik:
    gerçek gol artık pozisyon ağırlığına EKLENMİYOR.

    Pozisyon ağırlığını ÇARPIYOR.

    Böylece 2 gol atmış bir defans,
    2 gol atmış bir forvet kadar güçlü
    golcüye dönüşmüyor.
    """

    position_base = (
        match_players["position"]
        .map(GOAL_POSITION_BASE)
        .fillna(0.35)
        .astype(float)
    )

    performance_multiplier = (
        1.0
        + 0.90
        * match_players[
            "real_goals"
        ].astype(float)
        + 0.12
        * match_players[
            "real_assists"
        ].astype(float)
        + 0.10
        * np.sqrt(
            match_players[
                "simulated_goals"
            ].astype(float)
        )
    )

    # 90 dakika oynayan tam ağırlık.
    # 15 dakika oynayan çok daha düşük ağırlık.
    minutes_multiplier = (
        match_players[
            "match_minutes"
        ].astype(float)
        / 90.0
    )

    weights = (
        position_base
        * performance_multiplier
        * minutes_multiplier
    )

    return weights


def get_assist_weights(
    match_players,
):
    """
    Asist ağırlığı.
    """

    position_base = (
        match_players["position"]
        .map(ASSIST_POSITION_BASE)
        .fillna(0.8)
        .astype(float)
    )

    performance_multiplier = (
        1.0
        + 0.55
        * match_players[
            "real_assists"
        ].astype(float)
        + 0.08
        * match_players[
            "real_goals"
        ].astype(float)
        + 0.10
        * np.sqrt(
            match_players[
                "simulated_assists"
            ].astype(float)
        )
    )

    minutes_multiplier = (
        match_players[
            "match_minutes"
        ].astype(float)
        / 90.0
    )

    return (
        position_base
        * performance_multiplier
        * minutes_multiplier
    )


def choose_player(
    candidate_indices,
    weights,
    rng,
):
    """
    Ağırlıklı rastgele oyuncu seç.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    total = weights.sum()

    if total <= 0:
        probabilities = np.ones(
            len(weights)
        ) / len(weights)

    else:
        probabilities = (
            weights / total
        )

    chosen_index = rng.choice(
        candidate_indices,
        p=probabilities,
    )

    return chosen_index


def assign_team_goals(
    players,
    team,
    opponent,
    goal_count,
    date,
    rng,
    events,
    minutes_by_player,
):
    """
    Takımın attığı golleri maçta gerçekten
    dakika alan oyuncular arasında dağıtır.
    """

    participant_indices = list(
        minutes_by_player.keys()
    )

    match_players = players.loc[
        participant_indices
    ].copy()

    match_players[
        "match_minutes"
    ] = [
        minutes_by_player[index]
        for index in participant_indices
    ]

    for goal_number in range(
        1,
        goal_count + 1,
    ):

        # -----------------------------
        # GOLCÜ
        # -----------------------------

        goal_weights = get_goal_weights(
            match_players
        )

        scorer_index = choose_player(
            match_players.index.to_numpy(),
            goal_weights.to_numpy(),
            rng,
        )

        scorer_id = players.at[
            scorer_index,
            "player_id",
        ]

        scorer_name = players.at[
            scorer_index,
            "player_name",
        ]

        players.at[
            scorer_index,
            "simulated_goals",
        ] += 1

        players.at[
            scorer_index,
            "season_goals",
        ] += 1

        # Aynı maç içinde sonraki gol için
        # güncel değeri de yansıt.
        match_players.at[
            scorer_index,
            "simulated_goals",
        ] += 1

        # -----------------------------
        # ASİST
        # -----------------------------

        assist_id = None
        assist_name = None

        if (
            rng.random()
            < ASSIST_PROBABILITY
        ):

            assist_candidates = (
                match_players[
                    match_players[
                        "player_id"
                    ]
                    != scorer_id
                ].copy()
            )

            if not assist_candidates.empty:

                assist_weights = (
                    get_assist_weights(
                        assist_candidates
                    )
                )

                assist_index = (
                    choose_player(
                        assist_candidates
                        .index
                        .to_numpy(),
                        assist_weights
                        .to_numpy(),
                        rng,
                    )
                )

                assist_id = players.at[
                    assist_index,
                    "player_id",
                ]

                assist_name = players.at[
                    assist_index,
                    "player_name",
                ]

                players.at[
                    assist_index,
                    "simulated_assists",
                ] += 1

                players.at[
                    assist_index,
                    "season_assists",
                ] += 1

                match_players.at[
                    assist_index,
                    "simulated_assists",
                ] += 1

        events.append(
            {
                "date": date,
                "team": team,
                "opponent": opponent,
                "goal_number": goal_number,
                "scorer_id": scorer_id,
                "scorer": scorer_name,
                "assist_id": assist_id,
                "assist": assist_name,
            }
        )


def simulate_player_stats(
    simulated_matches,
    seed,
):
    players = load_player_data().copy()

    # Gerçek sezon istatistikleri.
    players["season_goals"] = (
        players["real_goals"].copy()
    )

    players["season_assists"] = (
        players["real_assists"].copy()
    )

    # Kalan simüle maçlardan gelecek veriler.
    players["simulated_goals"] = 0
    players["simulated_assists"] = 0

    players[
        "simulated_appearances"
    ] = 0

    players[
        "simulated_starts"
    ] = 0

    players[
        "simulated_minutes"
    ] = 0

    # Takım simülasyonundan ayrı random akışı.
    rng = np.random.default_rng(
        seed + 1
    )

    events = []

    matches = (
        simulated_matches
        .sort_values("date")
    )

    for _, match in matches.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_goals = int(
            match["fthg"]
        )

        away_goals = int(
            match["ftag"]
        )

        # -----------------------------
        # EV SAHİBİ KADROSU
        # -----------------------------

        (
            home_minutes,
            home_starters,
        ) = select_match_players(
            players,
            home_team,
            rng,
        )

        # -----------------------------
        # DEPLASMAN KADROSU
        # -----------------------------

        (
            away_minutes,
            away_starters,
        ) = select_match_players(
            players,
            away_team,
            rng,
        )

        # -----------------------------
        # APPEARANCE / START / MINUTES
        # -----------------------------

        for player_index, minutes in (
            home_minutes.items()
        ):
            players.at[
                player_index,
                "simulated_appearances",
            ] += 1

            players.at[
                player_index,
                "simulated_minutes",
            ] += minutes

        for player_index, minutes in (
            away_minutes.items()
        ):
            players.at[
                player_index,
                "simulated_appearances",
            ] += 1

            players.at[
                player_index,
                "simulated_minutes",
            ] += minutes

        players.loc[
            home_starters,
            "simulated_starts",
        ] += 1

        players.loc[
            away_starters,
            "simulated_starts",
        ] += 1

        # -----------------------------
        # GOLLER
        # -----------------------------

        assign_team_goals(
            players=players,
            team=home_team,
            opponent=away_team,
            goal_count=home_goals,
            date=match["date"],
            rng=rng,
            events=events,
            minutes_by_player=home_minutes,
        )

        assign_team_goals(
            players=players,
            team=away_team,
            opponent=home_team,
            goal_count=away_goals,
            date=match["date"],
            rng=rng,
            events=events,
            minutes_by_player=away_minutes,
        )

    events = pd.DataFrame(
        events
    )

    return players, events


def main():

    seed = None

    if len(sys.argv) > 1:
        seed = int(
            sys.argv[1]
        )

    # Takım sezonu.
    season_result = simulate_season(
        seed=seed
    )

    seed = season_result["seed"]

    # Oyuncu sezonu.
    players, events = (
        simulate_player_stats(
            season_result[
                "simulated_matches"
            ],
            seed,
        )
    )

    print()
    print("=" * 70)
    print(
        "2026/27 OYUNCU SİMÜLASYONU"
    )
    print("=" * 70)

    print()
    print(
        "Scenario seed:",
        seed,
    )

    # =============================
    # GOL KRALLIĞI
    # =============================

    print()
    print("GOL KRALLIĞI")
    print("------------")

    scorers = players.sort_values(
        [
            "season_goals",
            "season_assists",
        ],
        ascending=False,
    )

    print(
        scorers[
            [
                "player_name",
                "team",
                "position",
                "season_goals",
                "season_assists",
                "simulated_goals",
                "simulated_appearances",
                "simulated_starts",
                "simulated_minutes",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # =============================
    # ASİST KRALLIĞI
    # =============================

    print()
    print("ASİST KRALLIĞI")
    print("--------------")

    assisters = players.sort_values(
        [
            "season_assists",
            "season_goals",
        ],
        ascending=False,
    )

    print(
        assisters[
            [
                "player_name",
                "team",
                "position",
                "season_assists",
                "season_goals",
                "simulated_assists",
                "simulated_appearances",
                "simulated_starts",
                "simulated_minutes",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # =============================
    # KONTROL
    # =============================

    simulated_team_goals = int(
        (
            season_result[
                "simulated_matches"
            ]["fthg"]
            + season_result[
                "simulated_matches"
            ]["ftag"]
        ).sum()
    )

    simulated_player_goals = int(
        players[
            "simulated_goals"
        ].sum()
    )

    expected_player_minutes = (
        len(
            season_result[
                "simulated_matches"
            ]
        )
        * 2
        * 11
        * 90
    )

    actual_player_minutes = int(
        players[
            "simulated_minutes"
        ].sum()
    )

    print()
    print("KONTROL")
    print("-------")

    print(
        "Simüle takım golleri:",
        simulated_team_goals,
    )

    print(
        "Oyunculara dağıtılan goller:",
        simulated_player_goals,
    )

    print(
        "Gol dağılımı doğru:",
        simulated_team_goals
        == simulated_player_goals,
    )

    print(
        "Simüle gol olayı sayısı:",
        len(events),
    )

    print(
        "Beklenen toplam oyuncu dakikası:",
        expected_player_minutes,
    )

    print(
        "Gerçek simüle oyuncu dakikası:",
        actual_player_minutes,
    )

    print(
        "Dakika dağılımı doğru:",
        expected_player_minutes
        == actual_player_minutes,
    )


if __name__ == "__main__":
    main()