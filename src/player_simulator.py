import sys

import numpy as np
import pandas as pd

from player_data import load_player_data
from simulator import simulate_season


ASSIST_PROBABILITY = 0.72

REFERENCE_DATE = pd.Timestamp(
    "2026-09-19"
)


GOAL_POSITION_BASE = {
    "Offence": 2.7,
    "Midfield": 0.85,
    "Defence": 0.16,
    "Goalkeeper": 0.002,
}


ASSIST_POSITION_BASE = {
    "Offence": 2.2,
    "Midfield": 3.0,
    "Defence": 1.1,
    "Goalkeeper": 0.02,
}


STARTER_TARGETS = {
    "Goalkeeper": 1,
    "Defence": 4,
    "Midfield": 3,
    "Offence": 3,
}


BENCH_SIZE = 5


# ==========================================================
# OVERALL
# ==========================================================


def calculate_age(
    date_of_birth,
):
    if pd.isna(date_of_birth):
        return None

    birth_date = pd.to_datetime(
        date_of_birth,
        errors="coerce",
    )

    if pd.isna(birth_date):
        return None

    age = (
        REFERENCE_DATE.year
        - birth_date.year
    )

    if (
        REFERENCE_DATE.month,
        REFERENCE_DATE.day,
    ) < (
        birth_date.month,
        birth_date.day,
    ):
        age -= 1

    return age


def get_age_adjustment(age):
    """
    Overall = mevcut güç.

    Genç oyuncuların market value/potential
    konusunu daha sonra ayrı ele alacağız.
    """

    if age is None:
        return 0.0

    if age <= 18:
        return -1.0

    if age <= 21:
        return -0.5

    if age <= 29:
        return 0.0

    if age <= 32:
        return -0.5

    return -1.5


def add_player_overalls(
    players,
    team_ratings,
):
    """
    İlk overall modelimiz.

    Bu henüz gerçek oyuncu rating datası değil.
    Market value geldiğinde geliştirilecek.
    """

    players = players.copy()

    players["age"] = (
        players["date_of_birth"]
        .apply(calculate_age)
    )

    players["team_elo"] = (
        players["team"]
        .map(team_ratings)
        .fillna(1500.0)
        .astype(float)
    )

    # Takım seviyesini yaklaşık 70-85
    # overall tabanına dönüştürüyoruz.
    players["team_base_overall"] = (
        72.0
        + (
            players["team_elo"]
            - 1450.0
        )
        / 24.0
    ).clip(
        lower=70.0,
        upper=85.0,
    )

    position_bonus = (
        players["position"]
        .map(
            {
                "Offence": 1.0,
                "Midfield": 0.5,
                "Defence": 0.0,
                "Goalkeeper": 0.0,
            }
        )
        .fillna(0.0)
    )

    # Raw gol sayısını direkt eklemiyoruz.
    # sqrt ile aşırı büyümeyi engelliyoruz.
    goal_bonus = (
        2.3
        * np.sqrt(
            players["real_goals"]
            .astype(float)
        )
    )

    assist_bonus = (
        1.0
        * np.sqrt(
            players["real_assists"]
            .astype(float)
        )
    )

    age_adjustment = (
        players["age"]
        .apply(get_age_adjustment)
    )

    raw_overall = (
        players[
            "team_base_overall"
        ]
        + position_bonus
        + goal_bonus
        + assist_bonus
        + age_adjustment
    )

    players["overall"] = (
        raw_overall
        .round()
        .clip(
            lower=65,
            upper=92,
        )
        .astype(int)
    )

    return players


# ==========================================================
# SEÇİM AĞIRLIĞI
# ==========================================================


def get_selection_weights(
    players,
):
    """
    Overall ilk 11 seçiminde çok güçlü faktör.

    90 overall ile 75 overall arasında
    büyük fark yaratıyoruz.

    Ama simüle sezondaki form da
    küçük bir etki yapmaya devam ediyor.
    """

    overall = (
        players["overall"]
        .astype(float)
    )

    max_overall = overall.max()

    overall_strength = np.exp(
        (
            overall
            - max_overall
        )
        / 3.5
    )

    form_multiplier = (
        1.0
        + 0.10
        * players[
            "simulated_goals"
        ].astype(float)
        + 0.06
        * players[
            "simulated_assists"
        ].astype(float)
    )

    return (
        overall_strength
        * form_multiplier
    )


def weighted_sample(
    candidates,
    count,
    rng,
):
    if (
        candidates.empty
        or count <= 0
    ):
        return []

    count = min(
        count,
        len(candidates),
    )

    weights = get_selection_weights(
        candidates
    ).to_numpy(
        dtype=float
    )

    probabilities = (
        weights
        / weights.sum()
    )

    chosen = rng.choice(
        candidates.index.to_numpy(),
        size=count,
        replace=False,
        p=probabilities,
    )

    return chosen.tolist()


# ==========================================================
# MAÇ KADROSU
# ==========================================================


def select_match_players(
    players,
    team,
    rng,
):
    """
    Overall'a göre:

    11 starter
    +
    5 kişilik bench

    seçilir.

    Sonra daha düşük overall starterların
    değiştirilme ihtimali daha yüksek olur.
    """

    team_players = players[
        players["team"] == team
    ].copy()

    starters = []

    # -----------------------------
    # İLK 11
    # -----------------------------

    for (
        position,
        target,
    ) in STARTER_TARGETS.items():

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

        starters.extend(
            chosen
        )

    # Pozisyon eksikliği varsa
    # 11'e tamamla.
    if len(starters) < 11:

        remaining = team_players[
            ~team_players.index.isin(
                starters
            )
        ]

        missing = (
            11 - len(starters)
        )

        extra = weighted_sample(
            remaining,
            missing,
            rng,
        )

        starters.extend(
            extra
        )

    # -----------------------------
    # BENCH
    # -----------------------------

    remaining = team_players[
        ~team_players.index.isin(
            starters
        )
    ]

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

    # Kaç oyuncu değişikliği?
    number_of_subs = int(
        rng.integers(
            3,
            min(
                5,
                len(bench),
            )
            + 1,
        )
    )

    available_bench = (
        bench.copy()
    )

    replaced_starters = set()

    for _ in range(
        number_of_subs
    ):

        if not available_bench:
            break

        # -------------------------
        # YEDEKTEN GİRECEK OYUNCU
        # -------------------------

        bench_candidates = (
            players.loc[
                available_bench
            ]
        )

        bench_weights = (
            get_selection_weights(
                bench_candidates
            )
            .to_numpy(
                dtype=float
            )
        )

        bench_probabilities = (
            bench_weights
            / bench_weights.sum()
        )

        sub_index = int(
            rng.choice(
                bench_candidates
                .index
                .to_numpy(),
                p=bench_probabilities,
            )
        )

        available_bench.remove(
            sub_index
        )

        sub_position = players.at[
            sub_index,
            "position",
        ]

        sub_overall = players.at[
            sub_index,
            "overall",
        ]

        # -------------------------
        # ÇIKACAK STARTER
        # -------------------------

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
            candidates = (
                same_position
            )

        else:
            candidates = [
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

        if not candidates:
            continue

        starter_overalls = np.array(
            [
                players.at[
                    index,
                    "overall",
                ]
                for index in candidates
            ],
            dtype=float,
        )

        # Düşük overall oyuncunun
        # çıkma ihtimali daha yüksek.
        lowest = (
            starter_overalls.min()
        )

        outgoing_weights = np.exp(
            (
                starter_overalls.max()
                - starter_overalls
            )
            / 3.5
        )

        outgoing_probabilities = (
            outgoing_weights
            / outgoing_weights.sum()
        )

        starter_index = int(
            rng.choice(
                candidates,
                p=outgoing_probabilities,
            )
        )

        starter_overall = players.at[
            starter_index,
            "overall",
        ]

        # -------------------------
        # DEĞİŞİKLİK KARARI
        # -------------------------

        # Bench oyuncusu starterdan çok
        # daha kötüyse yıldızı sırf değişiklik
        # yapmak için çıkarmayalım.
        overall_difference = (
            sub_overall
            - starter_overall
        )

        if overall_difference <= -10:

            # Çok kötü bench alternatifi.
            # Sadece küçük ihtimalle değişiklik.
            if rng.random() > 0.12:
                continue

        elif overall_difference <= -6:

            if rng.random() > 0.30:
                continue

        replaced_starters.add(
            starter_index
        )

        # -------------------------
        # DEĞİŞİKLİK DAKİKASI
        # -------------------------

        # Düşük overall oyuncu daha erken
        # çıkabilir.

        earliest_minute = (
            55
            + max(
                0,
                int(
                    (
                        starter_overall
                        - 75
                    )
                    * 0.8
                ),
            )
        )

        earliest_minute = min(
            earliest_minute,
            80,
        )

        substitution_minute = int(
            rng.integers(
                earliest_minute,
                86,
            )
        )

        minutes[
            starter_index
        ] = substitution_minute

        minutes[
            sub_index
        ] = (
            90
            - substitution_minute
        )

    return (
        minutes,
        starters,
    )


# ==========================================================
# GOL / ASİST
# ==========================================================


def get_goal_weights(
    match_players,
):
    position_base = (
        match_players["position"]
        .map(GOAL_POSITION_BASE)
        .fillna(0.35)
        .astype(float)
    )

    # Gerçek sezonun ilk birkaç maçındaki form
    # önemli ama bütün yılı domine etmesin.
    real_form_multiplier = (
        1.0
        + 0.35
        * np.sqrt(
            match_players[
                "real_goals"
            ].astype(float)
        )
        + 0.12
        * np.sqrt(
            match_players[
                "real_assists"
            ].astype(float)
        )
    )

    # Alternatif evrende oluşan form.
    simulated_form_multiplier = (
        1.0
        + 0.08
        * np.sqrt(
            match_players[
                "simulated_goals"
            ].astype(float)
        )
    )

    # Overall farkını artık daha güçlü
    # şekilde gol yeteneğine yansıtıyoruz.
    overall_multiplier = np.exp(
        (
            match_players[
                "overall"
            ].astype(float)
            - 80.0
        )
        / 12.0
    )

    minutes_multiplier = (
        match_players[
            "match_minutes"
        ].astype(float)
        / 90.0
    )

    return (
        position_base
        * real_form_multiplier
        * simulated_form_multiplier
        * overall_multiplier
        * minutes_multiplier
    )

def get_assist_weights(
    match_players,
):
    position_base = (
        match_players["position"]
        .map(
            ASSIST_POSITION_BASE
        )
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

    overall_multiplier = (
        0.75
        + (
            match_players[
                "overall"
            ].astype(float)
            - 65.0
        )
        / 50.0
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
        * overall_multiplier
        * minutes_multiplier
    )


def choose_player(
    candidate_indices,
    weights,
    rng,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    total = weights.sum()

    if total <= 0:

        probabilities = (
            np.ones(
                len(weights)
            )
            / len(weights)
        )

    else:

        probabilities = (
            weights / total
        )

    return int(
        rng.choice(
            candidate_indices,
            p=probabilities,
        )
    )


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
    participant_indices = list(
        minutes_by_player.keys()
    )

    match_players = players.loc[
        participant_indices
    ].copy()

    match_players[
        "match_minutes"
    ] = [
        minutes_by_player[
            index
        ]
        for index
        in participant_indices
    ]

    for goal_number in range(
        1,
        goal_count + 1,
    ):

        # -------------------------
        # GOLCÜ
        # -------------------------

        goal_weights = (
            get_goal_weights(
                match_players
            )
        )

        scorer_index = choose_player(
            match_players
            .index
            .to_numpy(),
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

        match_players.at[
            scorer_index,
            "simulated_goals",
        ] += 1

        # -------------------------
        # ASİST
        # -------------------------

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
                ]
                .copy()
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


# ==========================================================
# OYUNCU SEZONU
# ==========================================================


def simulate_player_stats(
    simulated_matches,
    seed,
    starting_ratings,
):
    players = (
        load_player_data()
        .copy()
    )

    players = add_player_overalls(
        players,
        starting_ratings,
    )

    players["season_goals"] = (
        players["real_goals"]
        .copy()
    )

    players["season_assists"] = (
        players["real_assists"]
        .copy()
    )

    players[
        "simulated_goals"
    ] = 0

    players[
        "simulated_assists"
    ] = 0

    players[
        "simulated_appearances"
    ] = 0

    players[
        "simulated_starts"
    ] = 0

    players[
        "simulated_minutes"
    ] = 0

    rng = np.random.default_rng(
        seed + 1
    )

    events = []

    matches = (
        simulated_matches
        .sort_values("date")
    )

    for _, match in matches.iterrows():

        home_team = (
            match["home_team"]
        )

        away_team = (
            match["away_team"]
        )

        home_goals = int(
            match["fthg"]
        )

        away_goals = int(
            match["ftag"]
        )

        (
            home_minutes,
            home_starters,
        ) = select_match_players(
            players,
            home_team,
            rng,
        )

        (
            away_minutes,
            away_starters,
        ) = select_match_players(
            players,
            away_team,
            rng,
        )

        for (
            player_index,
            minutes,
        ) in home_minutes.items():

            players.at[
                player_index,
                "simulated_appearances",
            ] += 1

            players.at[
                player_index,
                "simulated_minutes",
            ] += minutes

        for (
            player_index,
            minutes,
        ) in away_minutes.items():

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

    return (
        players,
        events,
    )


# ==========================================================
# OUTPUT
# ==========================================================


def main():

    seed = None

    if len(sys.argv) > 1:
        seed = int(
            sys.argv[1]
        )

    season_result = (
        simulate_season(
            seed=seed
        )
    )

    seed = (
        season_result["seed"]
    )

    players, events = (
        simulate_player_stats(
            simulated_matches=(
                season_result[
                    "simulated_matches"
                ]
            ),
            seed=seed,
            starting_ratings=(
                season_result[
                    "starting_ratings"
                ]
            ),
        )
    )

    print()
    print("=" * 75)
    print(
        "2026/27 OYUNCU SİMÜLASYONU"
    )
    print("=" * 75)

    print()
    print(
        "Scenario seed:",
        seed,
    )

    # -----------------------------
    # OVERALL TOP 25
    # -----------------------------

    print()
    print("EN YÜKSEK OVERALL")
    print("-----------------")

    overall_table = (
        players
        .sort_values(
            [
                "overall",
                "real_goals",
            ],
            ascending=False,
        )
    )

    print(
        overall_table[
            [
                "player_name",
                "team",
                "position",
                "age",
                "overall",
                "real_goals",
                "real_assists",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    # -----------------------------
    # GOL KRALLIĞI
    # -----------------------------

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
                "overall",
                "season_goals",
                "season_assists",
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

    # -----------------------------
    # ASİST KRALLIĞI
    # -----------------------------

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
                "overall",
                "season_assists",
                "season_goals",
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

    # -----------------------------
    # KONTROL
    # -----------------------------

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

    print(
        "Gol olayı:",
        len(events),
    )


if __name__ == "__main__":
    main()