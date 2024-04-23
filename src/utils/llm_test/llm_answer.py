import argparse
import asyncio
import csv
import logging
import requests
import json
from time import sleep

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def summarize_content_yandex_gpt(
    prompt: str, model_name: str, temperature: float, folder_id: str, iam_token: str
) -> str:
    logging.info("Starting summarization with Yandex GPT")
    sleep(20)  # yandeggpt не любит слишком частые запросы

    messages = [
        {"role": "system", "text": prompt},
    ]
    data = {
        "modelUri": f"gpt://{folder_id}/{model_name}/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
        },
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}",
        "x-folder-id": folder_id,
    }
    response = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers=headers,
        data=json.dumps(data),
    )

    response_json = response.json()
    logging.debug(f"Response JSON from Yandex GPT: {response_json}")
    generated_text = response_json["result"]["alternatives"][0]["message"]["text"]
    logging.info("Summary was generated successfully")
    return generated_text


async def summarize_content_ollama(
    prompt: str,
    generator_host: str,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
) -> str:
    logging.info("Starting summarization with ollama")
    response = requests.post(
        generator_host,
        json={
            "model": model_name,
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": temperature, "repeat_penalty": repeat_penalty},
        },
    )
    logging.debug(f"Summary from ollama model: {response.json()['response']}")
    logging.info("Summary was generated successfully")
    return response.json()["response"]


async def summarize_content_gigachat(
    prompt: str,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
    auth_token: str,
) -> str:
    logging.info("Starting summarization with gigachat")
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "repetition_penalty": repeat_penalty,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, verify=False
    )
    logging.debug(f"Summary from gigachat model: {response.json()}")
    logging.info("Summary was generated successfully")

    return response.json()["choices"][0]["message"]["content"]


async def process_csv(
    input_filename: str,
    output_filename: str,
    prompt_template: str,
    model_name: str,
    host: str,
    temperature: float,
    repeat_penalty: float,
    token: str,
    folder_id: str,
    model_type: str,
) -> None:
    tasks = []
    with open(input_filename, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            context = f"{row['Документ1']}\n\n{row['Документ2']}\n\n{row['Документ3']}\n\n{row['Документ4']}"
            query = row["Вопрос"]
            prompt = prompt_template.format(context=context, query=query)
            logging.debug(f"Prompt for {model_type} model: {prompt}")

            if model_type == "ollama":
                tasks.append(
                    summarize_content_ollama(
                        prompt,
                        host,
                        model_name,
                        temperature,
                        repeat_penalty,
                    )
                )
            elif model_type == "yandex_gpt":
                logging.debug(f"Prompt for Yandex GPT: {prompt}")
                tasks.append(
                    summarize_content_yandex_gpt(
                        prompt,
                        model_name,
                        temperature,
                        folder_id,
                        token,
                    )
                )
            elif model_type == "gigachat":
                tasks.append(
                    summarize_content_gigachat(
                        prompt,
                        model_name,
                        temperature,
                        repeat_penalty,
                        token,
                    )
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

    results = await asyncio.gather(*tasks)

    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [
                "Вопрос",
                "Документ1",
                "Документ2",
                "Документ3",
                "Документ4",
                f"{model_name}_t{temperature}_rp{repeat_penalty}_pt_{prompt_template}",
            ]
        )
        for row, summary in zip(
            [
                row
                for row in csv.DictReader(
                    open(input_filename, mode="r", encoding="utf-8")
                )
            ],
            results,
        ):
            csv_writer.writerow(
                [
                    row["Вопрос"],
                    row["Документ1"],
                    row["Документ2"],
                    row["Документ3"],
                    row["Документ4"],
                    summary,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process CSV and use LLM for summarization."
    )
    parser.add_argument(
        "--input_filename", required=True, type=str, help="Input CSV filename"
    )
    parser.add_argument(
        "--output_filename", required=True, type=str, help="Output CSV filename"
    )
    parser.add_argument(
        "--prompt_template",
        required=True,
        type=str,
        help="Template for prompt creation",
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="Model name for LLM"
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
    parser.add_argument(
        "--token", type=str, required=False, help="IAM token for Yandex GPT"
    )
    parser.add_argument(
        "--folder_id", type=str, required=False, help="Folder ID for Yandex GPT"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        default="ollama",
        choices=["ollama", "yandex_gpt", "gigachat"],
        help="Type of the model to use",
    )

    args = parser.parse_args()

    asyncio.run(
        process_csv(
            args.input_filename,
            args.output_filename,
            args.prompt_template,
            args.model_name,
            args.host,
            args.temperature,
            args.repeat_penalty,
            args.token,
            args.folder_id,
            args.model_type,
        )
    )


if __name__ == "__main__":
    main()
