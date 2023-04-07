import asyncio
import csv

from src.scraper.scraper import (
    get_uid_from_url,
    collect_card_data,
    find_property,
    get_card_ability_text,
)


async def load_card_data():
    links = []
    with open("../../data/input_data/links.txt", "r") as f:
        for line in f:
            links.append(line.strip())

    card_ids = []
    for link in links:
        card_ids.append(get_uid_from_url(link))

    card_data = await collect_card_data(card_ids)

    new_card_data = []
    for new_card in card_data:
        card = new_card["card"]

        type = card["Text"]["ObjectType"]
        affinity = card["Text"]["Affinity"]
        rarity = card["Text"]["Rarity"]
        tribes = find_property(card["Text"]["Properties"], "TribalType").strip()
        realm = find_property(card["Text"]["Properties"], "Realm")
        ability_text = get_card_ability_text(card)
        cost = find_property(card["Text"]["Properties"], "IGOCost")
        hp = find_property(card["Text"]["Properties"], "HP")
        atk = find_property(card["Text"]["Properties"], "ATK")

        print(
            f"{type},{affinity},{rarity},{tribes},{realm},{ability_text},{cost},{hp},{atk}"
        )

        new_card_data.append(
            {
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

    with open(
        "../../data/input_data/new_card_data.csv",
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
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
        for card in new_card_data:
            writer.writerow(card)


def main():
    asyncio.run(load_card_data())


if __name__ == "__main__":
    main()
