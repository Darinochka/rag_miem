import argparse
import asyncio
import csv
import logging
import requests
import json
from time import sleep
import openai
from typing import List, Tuple, Any, Dict, Optional

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def summarize_content_yandex_gpt(
    prompt: str,
    system_prompt: Optional[str],
    model_name: str,
    temperature: float,
    folder_id: str,
    token: str,
    **kwargs: Any,
) -> str:
    logging.info("Starting summarization with Yandex GPT")
    sleep(20)  # yandeggpt не любит слишком частые запросы

    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

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
        "Authorization": f"Bearer {token}",
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


async def summarize_content_openai(
    prompt: str,
    system_prompt: Optional[str],
    client: openai.AsyncOpenAI,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
    **kwargs: Any,
) -> str:
    logging.info("Starting summarization with OpenAI")
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
    )
    logging.debug(f"Summary from OpenAI model: {completion.choices[0].message.content}")
    logging.info("Summary was generated successfully")
    return completion.choices[0].message.content


async def summarize_content_ollama(
    prompt: str,
    system_prompt: Optional[str],
    generator_host: str,
    model_name: str,
    **kwargs: Any,
) -> str:
    logging.info("Starting summarization with ollama")

    json_data = {
        "model": model_name,
        "stream": False,
        "prompt": prompt,
        "options": kwargs,
    }
    if system_prompt is not None:
        json_data["system"] = system_prompt

    response = requests.post(generator_host, json=json_data)
    logging.debug(f"Summary from ollama model: {response.json()['response']}")
    logging.info("Summary was generated successfully")
    return response.json()["response"]


async def summarize_content_gigachat(
    prompt: str,
    system_prompt: Optional[str],
    model_name: str,
    temperature: float,
    repeat_penalty: float,
    token: str,
    **kwargs: Any,
) -> str:
    logging.info("Starting summarization with gigachat")
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    payload = json.dumps(
        {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repeat_penalty,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, verify=False
    )
    logging.debug(f"Summary from gigachat model: {response.json()}")
    logging.info("Summary was generated successfully")

    return response.json()["choices"][0]["message"]["content"]


async def read_csv_and_generate_prompts(
    input_filename: str, prompt_template: str
) -> List[Tuple[Dict[str, Any], str]]:
    prompts = []
    with open(input_filename, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            context = f"{row['Документ1']}\n\n{row['Документ2']}\n\n{row['Документ3']}\n\n{row['Документ4']}"
            query = row["Вопрос"]
            prompt = prompt_template.format(context=context, query=query)
            prompts.append((row, prompt))
    return prompts


async def create_tasks(
    prompts: List[Tuple[Dict[str, Any], str]],
    system_prompt: Optional[str],
    model_name: str,
    host: str,
    model_type: str,
    **kwargs: Any,
) -> List[Any]:
    tasks = []
    for _, prompt in prompts:
        if model_type == "ollama":
            tasks.append(
                summarize_content_ollama(
                    prompt, system_prompt, host, model_name, **kwargs
                )
            )
        elif model_type == "yandex_gpt":
            tasks.append(
                summarize_content_yandex_gpt(
                    prompt, system_prompt, model_name, **kwargs
                )
            )
        elif model_type == "gigachat":
            tasks.append(
                summarize_content_gigachat(prompt, system_prompt, model_name, **kwargs)
            )
        elif model_type == "openai":
            client = openai.AsyncOpenAI(base_url=host, api_key=kwargs["token"])
            tasks.append(
                summarize_content_openai(
                    prompt, system_prompt, client, model_name, **kwargs
                )
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return tasks


async def write_results_to_csv(
    output_filename: str,
    model_name: str,
    prompts: List[Tuple[Dict[str, Any], str]],
    results: List[str],
    temperature: float,
    repeat_penalty: float,
    prompt_template: str,
) -> None:
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        header = [
            "Вопрос",
            "Документ1",
            "Документ2",
            "Документ3",
            "Документ4",
            f"{model_name}_t{temperature}_rp{repeat_penalty}_pt_{prompt_template}",
        ]
        csv_writer.writerow(header)
        for (row, _), summary in zip(prompts, results):
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


async def process_csv(
    input_filename: str,
    output_filename: str,
    prompt_template: str,
    system_prompt: Optional[str],
    model_name: str,
    host: str,
    temperature: float,
    repeat_penalty: float,
    token: str,
    folder_id: str,
    model_type: str,
) -> None:
    prompts = await read_csv_and_generate_prompts(input_filename, prompt_template)
    tasks = await create_tasks(
        prompts,
        system_prompt,
        model_name,
        host,
        model_type,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        token=token,
        folder_id=folder_id,
    )
    results = await asyncio.gather(*tasks)
    await write_results_to_csv(
        output_filename,
        model_name,
        prompts,
        results,
        temperature,
        repeat_penalty,
        prompt_template,
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
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt for the model",
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
        choices=["ollama", "yandex_gpt", "gigachat", "openai"],
        help="Type of the model to use",
    )

    args = parser.parse_args()

    asyncio.run(
        process_csv(
            args.input_filename,
            args.output_filename,
            args.prompt_template,
            args.system_prompt,
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
