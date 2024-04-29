import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
import argparse
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig
from typing import Optional, List, Any, Tuple, Dict
import httpx
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)


def get_parts(filename: str) -> List[Any]:
    parts = filename.split("_")
    embedding = parts[0]
    llm = parts[2]
    temperature = float(parts[4])
    repeat_penalty = float(parts[6])

    return [embedding, llm, temperature, repeat_penalty]


def create_models(
    model_name: str, api_key: str, openai_proxy_url: Optional[str]
) -> Tuple[ChatOpenAI, LangchainLLMWrapper, OpenAIEmbeddings]:
    gpt = ChatOpenAI(
        model_name=model_name,
        api_key=api_key,
        http_client=httpx.Client(proxy=openai_proxy_url),
    )
    ada_002 = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key,
        http_client=httpx.Client(proxy=openai_proxy_url),
    )
    gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)

    return gpt, gpt_wrapper, ada_002


def generate(
    model_name: str, api_key: str, openai_proxy_url: Optional[str], folder_path: str
) -> None:
    gpt, gpt_wrapper, ada_002 = create_models(model_name, api_key, openai_proxy_url)
    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = gpt_wrapper
        metric.embeddings = ada_002

    all_files = os.listdir(folder_path)
    for filename in all_files:
        if (
            filename.endswith(".csv")
            and not filename.endswith("result.csv")
            and f"{filename[:-4]}_result.csv" not in all_files
        ):
            logging.info(f"Processing {filename}")
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)[::-1]

            # Извлечение параметров из названия файла
            embedding, llm, temperature, repeat_penalty = get_parts(filename)
            # Последняя колонка - это ответы
            answer_col = data.columns[-1]

            data_samples = {
                "question": data["Вопрос"].tolist(),
                "answer": data[answer_col].tolist(),
                "contexts": data[
                    ["Документ1", "Документ2", "Документ3", "Документ4"]
                ].values.tolist(),
            }
            logging.info(f"Processing {len(data_samples['question'])} questions")
            dataset = Dataset.from_dict(data_samples)

            score = evaluate(
                dataset=dataset,
                llm=gpt,
                embeddings=ada_002,
                metrics=metrics,
                run_config=RunConfig(timeout=150),
            )

            score_pd = score.to_pandas()
            print(
                f"{embedding},{llm},{temperature},{repeat_penalty},{score_pd['faithfulness'].mean()},{score_pd['answer_relevancy'].mean()}"
            )
            score_pd.to_csv(os.path.join(folder_path, f"{filename[:-4]}_result.csv"))


def preprocess_contexts(contexts: str) -> List[str]:
    new_contexts = contexts.split("\n")
    # удаляем [ и ]
    new_contexts[0] = new_contexts[0][1:]
    new_contexts[-1] = new_contexts[-1][:-1]
    # удаляем кавычки вокруг контекстов
    new_contexts = [ctx[1:-1] for ctx in new_contexts]
    return new_contexts


def calculate_faithfulness(
    row: Dict[str, Any], gpt: ChatOpenAI, ada_002: OpenAIEmbeddings
) -> float:
    if np.isnan(row["faithfulness"]):
        data_samples = {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [preprocess_contexts(row["contexts"])],
        }
        print(data_samples)
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset=dataset,
            llm=gpt,
            embeddings=ada_002,
            metrics=[faithfulness],
            run_config=RunConfig(timeout=150),
        )
        print(score)
        return score["faithfulness"]
    else:
        return row["faithfulness"]


def fillna_faithfulness(
    model_name: str, api_key: str, openai_proxy_url: str, folder_path: str
) -> None:
    gpt, _, ada_002 = create_models(model_name, api_key, openai_proxy_url)

    for filename in os.listdir(folder_path):
        if filename.endswith("result.csv"):
            logging.info(f"Processing {filename}")
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            data["faithfulness"] = data.apply(
                calculate_faithfulness, axis=1, raw=False, args=(gpt, ada_002)
            )
            data.to_csv(os.path.join("data/results_metrics/", filename), index=False)


def evaluation(folder_path: str) -> None:
    for filename in os.listdir(folder_path):
        if filename.endswith("result.csv"):
            file_path = os.path.join(folder_path, filename)
            embedding, llm, temperature, repeat_penalty = get_parts(filename)
            data = pd.read_csv(file_path)
            data = data.fillna(0)
            print(
                f"{embedding},{llm},{temperature},{repeat_penalty},{data['faithfulness'].mean()},{data['answer_relevancy'].mean()}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate text answers for faithfulness and relevancy."
    )
    parser.add_argument(
        "--api_key", type=str, help="OpenAI API key for accessing the models"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to the folder containing CSV files for evaluation",
    )
    parser.add_argument(
        "--openai_proxy_url",
        default=None,
        type=str,
        help="URL of the OpenAI proxy server",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo-0125",
        type=str,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--mode",
        default="gen+eval",
        choices=["gen", "eval", "gen+eval"],
    )
    args = parser.parse_args()

    if args.mode == "gen+eval":
        generate(args.model_name, args.api_key, args.openai_proxy_url, args.folder_path)
        evaluation(args.folder_path)
    elif args.mode == "eval":
        evaluation(args.folder_path)
    elif args.mode == "gen":
        generate(args.model_name, args.api_key, args.openai_proxy_url, args.folder_path)
    else:
        raise ValueError(f"Unknown mode {args.mode}")
