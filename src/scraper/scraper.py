import asyncio
import csv
from psycopg2 import Error
import re

import aiohttp as aiohttp
import requests

from src.scraper.database import connect_to_database, close_connection

api_card_url = "https://server.collective.gg/api/card/"


def get_uid_from_url(url):
    uid = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", url
    ).group(0)
    return uid


def get_card_info(card_id):
    return requests.get(api_card_url + card_id).json()["card"]


async def fetch_card(session, card_id):
    url = f"https://server.collective.gg/api/card/{card_id}"
    async with session.get(url) as response:
        card_data = await response.json()
        return card_data


async def collect_card_data(card_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_card(session, card_id) for card_id in card_ids]
        card_data = await asyncio.gather(*tasks)
        return card_data


def get_card_ability_text(card_json):
    ability_text = ""

    if "Properties" in card_json["Text"]["PlayAbility"]:
        for index, node in enumerate(card_json["Text"]["PlayAbility"]["Properties"]):
            if node["Symbol"]["Name"] == "AbilityText":
                ability_text += card_json["Text"]["PlayAbility"]["Properties"][index][
                    "Expression"
                ]["Value"]

    if "Abilities" in card_json["Text"]:
        for ability_index, ability in enumerate(card_json["Text"]["Abilities"]):
            if "Properties" in ability:
                for index, node in enumerate(
                    card_json["Text"]["Abilities"][ability_index]["Properties"]
                ):
                    if node["Symbol"]["Name"] == "AbilityText":
                        ability_text += (
                            " "
                            + card_json["Text"]["Abilities"][ability_index][
                                "Properties"
                            ][index]["Expression"]["Value"]
                        )

    return ability_text.strip()


def find_property(properties, target):
    for prop in properties:
        if prop["Symbol"]["Name"] == target:
            return prop["Expression"]["Value"]
    return ""


async def fetch_posts():
    post_data = []

    connection, cursor = connect_to_database()

    try:
        cursor.execute(
            """
            SELECT object_printing_id, subreddit_votes, dt_subreddit_submitted
            FROM subreddit_cards
            WHERE title LIKE '[Card]%%'
        """
        )
        records = cursor.fetchall()
        for record in records:
            post_data.append(
                {
                    "uid": record[0],
                    "score": record[1],
                    "created_utc": int(record[2].timestamp()),
                }
            )
    except (Exception, Error) as error:
        print("Error while fetching data from PostgreSQL", error)
    finally:
        close_connection(connection)

    posts = []
    card_ids = []
    for post in post_data:
        card_ids.append(post["uid"])
        print(f"{post['uid']} {post['score']}")

    card_data = await collect_card_data(card_ids)

    for post, card_d in zip(post_data, card_data):
        if "card" not in card_d:
            continue
        card = card_d["card"]

        votes = post["score"]
        timestamp = post["created_utc"]
        name = card["Text"]["Name"].strip()
        type = card["Text"]["ObjectType"]
        affinity = card["Text"]["Affinity"]
        rarity = card["Text"]["Rarity"]
        tribes = find_property(card["Text"]["Properties"], "TribalType").strip()
        realm = find_property(card["Text"]["Properties"], "Realm")
        ability_text = get_card_ability_text(card)
        cost = find_property(card["Text"]["Properties"], "IGOCost")
        hp = find_property(card["Text"]["Properties"], "HP")
        atk = find_property(card["Text"]["Properties"], "ATK")

        posts.append(
            {
                "votes": votes,
                "timestamp": timestamp,
                "name": name,
                "type": type,
                "affinity": affinity,
                "rarity": rarity,
                "tribes": tribes,
                "realm": realm,
                "ability_text": ability_text,
                "cost": cost,
                "hp": hp,
                "atk": atk,
            }
        )

    return posts


async def main():
    posts = await fetch_posts()

    with open(
        "../../data/raw_data/cards_raw.csv", mode="w", encoding="utf-8", newline=""
    ) as csv_file:
        fieldnames = [
            "votes",
            "timestamp",
            "name",
            "type",
            "affinity",
            "rarity",
            "tribes",
            "realm",
            "ability_text",
            "cost",
            "hp",
            "atk",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for post in posts:
            writer.writerow(
                {
                    "votes": post["votes"],
                    "timestamp": post["timestamp"],
                    "name": post["name"],
                    "type": post["type"],
                    "affinity": post["affinity"],
                    "rarity": post["rarity"],
                    "tribes": post["tribes"],
                    "realm": post["realm"],
                    "ability_text": post["ability_text"],
                    "cost": post["cost"],
                    "hp": post["hp"],
                    "atk": post["atk"],
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
    print("Done fetching cards!")
