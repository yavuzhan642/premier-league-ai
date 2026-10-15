import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = (
    "https://www.ea.com/games/"
    "ea-sports-fc/ratings/"
    "leagues-ratings/"
    "premier-league/13"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "fc27_premier_league_ratings.csv"
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept-Language": (
        "en-US,en;q=0.9"
    ),
}


def extract_number(text):
    """
    'OVR 91' -> 91
    'PAC 87' -> 87
    """

    if text is None:
        return None

    numbers = re.findall(
        r"\d+",
        str(text),
    )

    if not numbers:
        return None

    return int(
        numbers[-1]
    )

def clean_player_name(name):
    """
    '#3 Erling Haaland ST'
    -> 'Erling Haaland'
    """

    name = str(name).strip()

    name = re.sub(
        r"^#\d+\s*",
        "",
        name,
    )

    name = re.sub(
        r"\s*Vote.*$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # EA bazen pozisyonu oyuncu adının
    # sonuna yapıştırıyor.
    position_codes = (
        "GK|CB|LB|RB|LWB|RWB|"
        "CDM|CM|CAM|LM|RM|"
        "LW|RW|CF|ST"
    )

    name = re.sub(
        rf"\s+({position_codes})$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name.strip()

def get_cell_image_name(
    cell,
):
    """
    Team / nationality hücresinde
    bazen text yerine image alt/title var.
    """

    if cell is None:
        return None

    text = cell.get_text(
        " ",
        strip=True,
    )

    if text:
        return text

    image = cell.find("img")

    if image:
        for attribute in [
            "alt",
            "title",
        ]:
            value = image.get(
                attribute
            )

            if value:
                return value.strip()

    return None


def fetch_page(
    session,
    page,
):
    response = session.get(
        BASE_URL,
        params={
            "page": page,
        },
        headers=HEADERS,
        timeout=30,
    )

    print(
        f"EA page {page} status:",
        response.status_code,
    )

    response.raise_for_status()

    return response.text


def parse_page(
    html,
):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    rows = []

    # EA sayfasındaki bütün table row'ları.
    for tr in soup.find_all("tr"):

        player_link = tr.find(
            "a",
            href=lambda href: (
                href
                and
                "/ratings/player-ratings/"
                in href
            ),
        )

        if player_link is None:
            continue

        cells = tr.find_all(
            "td"
        )

        # Beklediğimiz yapı:
        #
        # rank
        # player
        # nationality
        # team
        # position
        # overall
        # pace
        # shooting
        # passing
        # dribbling
        # defending
        # physical
        #
        if len(cells) < 12:
            continue

        cell_texts = [
            cell.get_text(
                " ",
                strip=True,
            )
            for cell in cells
        ]

        rank = extract_number(
            cell_texts[0]
        )

        player_name = (
            clean_player_name(
                player_link.get_text(
                    " ",
                    strip=True,
                )
            )
        )

        nationality = (
            get_cell_image_name(
                cells[2]
            )
        )

        team = (
            get_cell_image_name(
                cells[3]
            )
        )

        position = (
            cell_texts[4]
            .strip()
        )

        overall = extract_number(
            cell_texts[5]
        )

        pace = extract_number(
            cell_texts[6]
        )

        shooting = extract_number(
            cell_texts[7]
        )

        passing = extract_number(
            cell_texts[8]
        )

        dribbling = extract_number(
            cell_texts[9]
        )

        defending = extract_number(
            cell_texts[10]
        )

        physical = extract_number(
            cell_texts[11]
        )

        player_url = urljoin(
            BASE_URL,
            player_link.get("href"),
        )

        # Gerçek ratings satırı olduğundan
        # emin ol.
        if (
            not player_name
            or overall is None
        ):
            continue

        rows.append(
            {
                "rank": rank,
                "player_name": (
                    player_name
                ),
                "nationality": (
                    nationality
                ),
                "team": team,
                "fc_position": (
                    position
                ),
                "overall": overall,
                "pace": pace,
                "shooting": (
                    shooting
                ),
                "passing": passing,
                "dribbling": (
                    dribbling
                ),
                "defending": (
                    defending
                ),
                "physical": physical,
                "player_url": (
                    player_url
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def load_fc27_premier_league(
    max_pages=40,
):
    """
    EA Premier League ratings
    sayfalarının tamamını dolaşır.

    Yeni oyuncu gelmeyen sayfada durur.
    """

    session = requests.Session()

    all_pages = []

    seen_players = set()

    for page in range(
        1,
        max_pages + 1,
    ):
        html = fetch_page(
            session,
            page,
        )

        page_data = parse_page(
            html
        )

        if page_data.empty:
            print(
                f"Page {page}: "
                "oyuncu bulunamadı."
            )

            break

        # Aynı sayfanın tekrar dönmesi
        # ihtimaline karşı kontrol.
        new_rows = []

        for _, row in (
            page_data.iterrows()
        ):
            key = (
                row["player_url"]
                or (
                    row["player_name"],
                    row["overall"],
                )
            )

            if key in seen_players:
                continue

            seen_players.add(
                key
            )

            new_rows.append(
                row
            )

        if not new_rows:
            print(
                f"Page {page}: "
                "yeni oyuncu yok, duruyoruz."
            )

            break

        new_df = pd.DataFrame(
            new_rows
        )

        print(
            f"Page {page}: "
            f"{len(new_df)} yeni oyuncu"
        )

        all_pages.append(
            new_df
        )

        # EA'ya saniyede onlarca istek
        # göndermeyelim.
        time.sleep(1)

    if not all_pages:
        raise RuntimeError(
            "EA sayfasından oyuncu "
            "verisi okunamadı."
        )

    ratings = pd.concat(
        all_pages,
        ignore_index=True,
    )

    ratings = (
        ratings
        .drop_duplicates(
            subset=[
                "player_url",
            ]
        )
    )

    ratings = (
        ratings
        .sort_values(
            [
                "overall",
                "rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return ratings


def save_ratings(
    ratings,
):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ratings.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Kaydedildi:",
        OUTPUT_PATH,
    )


def main():
    ratings = (
        load_fc27_premier_league()
    )

    save_ratings(
        ratings
    )

    print()
    print(
        "FC 27 PREMIER LEAGUE"
    )
    print(
        "===================="
    )

    print()
    print(
        "Toplam oyuncu:",
        len(ratings),
    )

    print()
    print(
        ratings[
            [
                "player_name",
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
    print(
        "HAALAND KONTROLÜ"
    )
    print(
        "----------------"
    )

    haaland = ratings[
        ratings[
            "player_name"
        ]
        .str.contains(
            "Haaland",
            case=False,
            na=False,
        )
    ]

    if haaland.empty:
        print(
            "Haaland bulunamadı!"
        )

    else:
        print(
            haaland[
                [
                    "player_name",
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
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()