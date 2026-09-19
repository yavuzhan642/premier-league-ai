import sys

import numpy as np
import pandas as pd

from player_data import load_player_data
from simulator import simulate_season


ASSIST_PROBABILITY = 0.72


GOAL_POSITION_BASE = {
    "Offence": 2.5,
    "Midfield": 0.8,
    "Defence": 0.18,
    "Goalkeeper": 0.002,
}


ASSIST_POSITION_BASE = {
    "Offence": 2.2,
    "Midfield": 3.0,
    "Defence": 1.2,
    "Goalkeeper": 0.03,
}

def get_goal_weights(team_players):
    position_weight = (
        team_players["position"]
        .map(GOAL_POSITION_BASE)
        .fillna(0.4)
        .astype(float)
    )

    real_goal_boost = (
        3.0
        * team_players["real_goals"]
        .astype(float)
    )

    real_assist_boost = (
        0.35
        * team_players["real_assists"]
        .astype(float)
    )

    breakout_boost = (
        0.45
        * np.sqrt(
            team_players["simulated_goals"]
            .astype(float)
        )
    )

    weights = (
        position_weight
        + real_goal_boost
        + real_assist_boost
        + breakout_boost
    )

    return weights

def get_assist_weights(team_players):
    position_weight = (
        team_players["position"]
        .map(ASSIST_POSITION_BASE)
        .fillna(1.0)
        .astype(float)
    )

    form_multiplier = (
        1.0
        + 0.30
        * np.sqrt(
            team_players["season_assists"]
            .astype(float)
        )
        + 0.05
        * np.sqrt(
            team_players["season_goals"]
            .astype(float)
        )
    )

    return position_weight * form_multiplier


def choose_player(
    players,
    candidate_indices,
    weights,
    rng,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    probabilities = (
        weights / weights.sum()
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
):
    for goal_number in range(
        1,
        goal_count + 1,
    ):
        team_players = players[
            players["team"] == team
        ]

        goal_weights = get_goal_weights(
            team_players
        )

        scorer_index = choose_player(
            players,
            team_players.index.to_numpy(),
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

        assist_name = None
        assist_id = None

        # Her golün asisti olmak zorunda değil.
        if rng.random() < ASSIST_PROBABILITY:
            assist_candidates = players[
                (players["team"] == team)
                & (
                    players["player_id"]
                    != scorer_id
                )
            ]

            if not assist_candidates.empty:
                assist_weights = (
                    get_assist_weights(
                        assist_candidates
                    )
                )

                assist_index = choose_player(
                    players,
                    assist_candidates
                    .index
                    .to_numpy(),
                    assist_weights.to_numpy(),
                    rng,
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

    # Gerçek sezon istatistiği başlangıç noktası.
    players["season_goals"] = (
        players["real_goals"].copy()
    )

    players["season_assists"] = (
        players["real_assists"].copy()
    )

    players["simulated_goals"] = 0
    players["simulated_assists"] = 0

    # Maç simülasyonundan farklı RNG akışı.
    # Aynı scenario seed yine aynı oyuncu
    # sezonunu üretir.
    rng = np.random.default_rng(
        seed + 1
    )

    events = []

    matches = simulated_matches.sort_values(
        "date"
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

        assign_team_goals(
            players=players,
            team=home_team,
            opponent=away_team,
            goal_count=home_goals,
            date=match["date"],
            rng=rng,
            events=events,
        )

        assign_team_goals(
            players=players,
            team=away_team,
            opponent=home_team,
            goal_count=away_goals,
            date=match["date"],
            rng=rng,
            events=events,
        )

    events = pd.DataFrame(events)

    return players, events


def main():
    seed = None

    if len(sys.argv) > 1:
        seed = int(
            sys.argv[1]
        )

    season_result = simulate_season(
        seed=seed
    )

    seed = season_result["seed"]

    players, events = simulate_player_stats(
        season_result[
            "simulated_matches"
        ],
        seed,
    )

    print()
    print("=" * 60)
    print("2026/27 OYUNCU SİMÜLASYONU")
    print("=" * 60)

    print()
    print(
        "Scenario seed:",
        seed,
    )

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
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

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
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # KONTROL
    # --------------------------------

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


if __name__ == "__main__":
    main()
