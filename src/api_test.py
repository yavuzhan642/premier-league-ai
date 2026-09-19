import os

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

URL = "https://api.football-data.org/v4/competitions/PL/matches"

headers = {
    "X-Auth-Token": API_KEY,
}

params = {
    "season": 2026,
}


response = requests.get(
    URL,
    headers=headers,
    params=params,
    timeout=30,
)

print("Status code:", response.status_code)

print(
    "Kalan istek hakkı:",
    response.headers.get("X-Requests-Available-Minute"),
)

print(
    "Rate limit reset:",
    response.headers.get("X-RequestCounter-Reset"),
)

if response.ok:
    data = response.json()

    print("Toplam maç:", len(data["matches"]))

    finished = [
        match
        for match in data["matches"]
        if match["status"] == "FINISHED"
    ]

    print("Oynanmış maç:", len(finished))

    print("\nSon 5 oynanmış maç:")

    for match in finished[-5:]:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        print(
            f"{home_goals}-{away_goals} | "
            f"{home} vs {away}"
        )

else:
    print(response.text)