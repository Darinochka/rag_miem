import argparse
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


def parse_links(input_file: str, output_file: str) -> None:
    target_classes = [
        "l-extra js-mobile_menu_content is-desktop",
        "person-caption",
        "g-ul g-list small",
        "g-ul g-list",
        "g-ul g-list small person-employment-addition",
        "main-list large main-list-language-knowledge-level",
        "main-list large",
        "main-list person-extra-indent",
    ]

    valid_url_patterns = [
        re.compile(r"^http://www\.hse\.ru/org/persons/\d+$"),
        re.compile(r"^http://www\.hse\.ru/staff/[\w\d]+$"),
    ]
    data = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            links = re.findall(r"\[(.*?)\]\((.*?)\)", line)

            if len(links) != 1:
                continue

            name, url = links[0]
            if not any(pattern.match(url) for pattern in valid_url_patterns):
                continue

            print(f"Processing {url}")

            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            extracted_texts = []
            for target_class in target_classes:
                elements = soup.find_all(class_=target_class)
                for element in elements:
                    extracted_texts.append(element.get_text(" ", strip=True))

            data.append({"source": url, "text": " ".join(extracted_texts)})

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Парсинг ссылок и сохранение данных в CSV."
    )
    parser.add_argument(
        "--input_file", type=str, help="Путь к входному файлу со списком ссылок."
    )
    parser.add_argument("--output_file", type=str, help="Путь к выходному файлу CSV.")

    args = parser.parse_args()

    parse_links(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
