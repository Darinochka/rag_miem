import requests
import pandas as pd
import argparse
from typing import Any


def retrieve_documents(question: str) -> Any:
    url = "http://localhost:8000/invoke"
    inputs = {"input": question}
    response = requests.post(url, json=inputs)
    if response.status_code == 200:
        return response.json()["output"]
    else:
        raise Exception(f"Ошибка: {response.status_code}")


def process_csv(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    results = []
    for question in df.iloc[:, 0]:
        documents = retrieve_documents(question)
        documents_contents = [doc["page_content"] for doc in documents]
        results.append([question] + documents_contents)
    results_df = pd.DataFrame(
        results, columns=["Вопрос", "Документ1", "Документ2", "Документ3", "Документ4"]
    )
    results_df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка вопросов и получение документов."
    )
    parser.add_argument(
        "--input_csv", required=True, help="Путь к входному файлу CSV с вопросами"
    )
    parser.add_argument(
        "--output_csv", required=True, help="Путь для сохранения результатов в CSV файл"
    )

    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
