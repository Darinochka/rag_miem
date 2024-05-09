import argparse
import requests
from bs4 import BeautifulSoup
import csv
from typing import Any, List


def parse_header_content(header_content: Any) -> str:
    texts = []

    # Извлечение текста из обычных div элементов
    divs = header_content.find_all("div", recursive=False)
    for div in divs:
        if "class" not in div.attrs or not any(
            cls in div["class"] for cls in ["with-indent2", "with-indent5", "menu"]
        ):
            texts.append(div.get_text(strip=True))

    # Извлечение текста из first_child
    first_child = header_content.find("p", class_="first_child ")
    if first_child:
        texts.append(first_child.get_text(strip=True))

    # Извлечение текста из first_child
    first_child = header_content.find("p", class_="last_child ")
    if first_child:
        texts.append(first_child.get_text(strip=True))

    # Извлечение текста из first_child
    first_child = header_content.find(class_="first_child")
    if first_child:
        texts.append(first_child.get_text(strip=True))

    # Извлечение текста из first_child2
    first_child = header_content.find(
        class_="with-indent lead-in _builder builder--text"
    )
    if first_child:
        texts.append(first_child.get_text(strip=True))

    return ". ".join(filter(None, texts))


def parse_html(url: str, session: requests.Session) -> str:
    response = session.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    header_content = soup.find(class_="header-content")
    header_text = parse_header_content(header_content) if header_content else ""
    first_child = header_content.find(class_="first_child") if header_content else None
    program_name = first_child.text.strip() if first_child else ""

    # Парсинг b-row с разделением текстов дочерних элементов
    b_row = soup.find_all(
        class_="with-indent2 b-row__item b-row__item--size_3 b-row__item--t6"
    )
    b_row_texts: List[str] = []
    if b_row:
        for item in b_row:
            b_row_texts.extend(
                set([elem.get_text(strip=True) for elem in item.find_all()])
            )
        b_row_text = "\n".join(filter(None, b_row_texts))
    else:
        b_row_text = ""

    # Парсинг incut_items с разделением текстов дочерних элементов и добавлением program_name
    incut_items = soup.find_all(class_="incut foldable_block__item")
    texts = []
    for item in incut_items:
        item_header_list = [
            elem.get_text(strip=True)
            for elem in item.find_all("span", "_link _link--pseudo")
        ]
        item_header = ". ".join(filter(None, item_header_list))

        item_texts = [elem.get_text(strip=True) for elem in item.find_all("p")]
        print(item_texts)
        item_text = ". ".join(filter(None, item_texts))

        # Добавляем program_name к первому элементу incut_items, если это необходимо
        item_text = f"[{program_name}] {item_header}\n{item_text}"

        texts.append(item_text)

    # Соединяем все полученные тексты
    text = "\n".join(texts)
    full_text = f"{header_text}\n{b_row_text}\n{text}"

    return full_text


def main(input_filename: str, output_filename: str) -> None:
    session = requests.Session()
    results = []
    with open(input_filename, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    for url in urls:
        text = parse_html(url, session)
        results.append([url, text])

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source", "text"])
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename", type=str, required=True)
    parser.add_argument("--output-filename", type=str, required=True)
    args = parser.parse_args()

    main(args.input_filename, args.output_filename)
