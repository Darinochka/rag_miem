import argparse
import asyncio
import csv
import logging
import requests
from typing import Dict, Any, List

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def summarize_content_ollama(
    prompt: str,
    generator_host: str,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
) -> Dict[Any, Any]:
    response = requests.post(
        generator_host,
        json={
            "model": model_name,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": temperature, "repeat_penalty": repeat_penalty},
        },
    )
    logging.debug(f"Summary from ollama model: {response.json()}")
    return response.json()


async def calc_avg_tps(
    input_filename: str,
    prompt_template: str,
    model_names: List[str],
    host: str,
    temperature: float,
    repeat_penalty: float,
) -> None:
    for model_name in model_names:
        tasks = []
        with open(input_filename, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                context = f"{row['Документ1']}\n\n{row['Документ2']}\n\n{row['Документ3']}\n\n{row['Документ4']}"
                query = row["Вопрос"]
                prompt = prompt_template.format(context=context, query=query)
                logging.debug(f"Prompt for {model_name} model: {prompt}")

                tasks.append(
                    summarize_content_ollama(
                        prompt,
                        host,
                        model_name,
                        temperature,
                        repeat_penalty,
                    )
                )

        results = await asyncio.gather(*tasks)

        total_tps = 0
        for response in results:
            total_tps += response["eval_count"] / response["eval_duration"]

        print(
            f"{model_name},{repeat_penalty},{temperature},{total_tps / len(results) * 10**9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process CSV and use LLM for summarization."
    )
    parser.add_argument(
        "--input_filename", required=True, type=str, help="Input CSV filename"
    )
    parser.add_argument(
        "--prompt_template",
        required=True,
        type=str,
        help="Template for prompt creation",
    )
    parser.add_argument(
        "--model_names", required=True, type=str, nargs="+", help="Model names for LLM"
    )
    parser.add_argument(
        "--host", required=True, type=str, help="Host URL for the model"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature setting for generation",
    )
    parser.add_argument(
        "--repeat_penalty", type=float, required=True, help="repeat penalty to generate"
    )

    args = parser.parse_args()

    asyncio.run(
        calc_avg_tps(
            args.input_filename,
            args.prompt_template,
            args.model_names,
            args.host,
            args.temperature,
            args.repeat_penalty,
        )
    )


if __name__ == "__main__":
    main()
