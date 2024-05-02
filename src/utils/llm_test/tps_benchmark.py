import argparse
import asyncio
import csv
import logging
import requests
from typing import Dict, Any, List, Optional
import openai

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def summarize_content_ollama(
    prompt: str,
    system_prompt: Optional[str],
    generator_host: str,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
) -> Dict[Any, Any]:
    json_data = {
        "model": model_name,
        "stream": False,
        "prompt": prompt,
        "options": {"temperature": temperature, "repeat_penalty": repeat_penalty},
    }
    if system_prompt is not None:
        json_data["system"] = system_prompt

    response = requests.post(generator_host, json=json_data)
    logging.debug(f"Summary from ollama model: {response.json()}")
    return response.json()


async def summarize_content_openai(
    prompt: str,
    system_prompt: Optional[str],
    client: openai.AsyncOpenAI,
    model_name: str,
    temperature: float,
    repeat_penalty: float,
) -> Dict[Any, Any]:
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
        frequency_penalty=repeat_penalty - 1,
    )
    logging.debug(f"Summary from OpenAI model: {completion.choices[0].message.content}")
    return completion


async def tps_ollama(results: List[Dict[str, Any]]) -> float:
    total_tps = 0
    for response in results:
        total_tps += response["eval_count"] / response["eval_duration"]
    return total_tps / len(results) * 10**9


async def tps_openai(host: str) -> Optional[float]:
    headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
    host += "/metrics"
    response = requests.get(host, headers=headers)
    lines = response.text.split("\n")
    predicted_tokens_seconds = None
    for line in lines:
        if "llamacpp:predicted_tokens_seconds" in line:
            predicted_tokens_seconds = float(line.split()[-1])
    return predicted_tokens_seconds


async def process_model(
    model_name: str,
    prompts: List[str],
    system_prompt: Optional[str],
    host: str,
    model_type: str,
    **kwargs: Any,
) -> Optional[float]:
    tasks = []
    if model_type == "ollama":
        tasks = [
            summarize_content_ollama(prompt, system_prompt, host, model_name, **kwargs)
            for prompt in prompts
        ]
    elif model_type == "openai":
        client = openai.AsyncOpenAI(base_url=host + "/v1", api_key="sk-...")
        tasks = [
            summarize_content_openai(
                prompt, system_prompt, client, model_name, **kwargs
            )
            for prompt in prompts
        ]

    results = await asyncio.gather(*tasks)
    if model_type == "ollama":
        return await tps_ollama(results)
    elif model_type == "openai":
        return await tps_openai(host)
    else:
        return None


async def generate_prompts(input_filename: str, prompt_template: str) -> List[str]:
    with open(input_filename, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        return [
            prompt_template.format(
                context=f"{row['Документ1']}\n\n{row['Документ2']}\n\n{row['Документ3']}\n\n{row['Документ4']}",
                query=row["Вопрос"],
            )
            for row in csv_reader
        ]


async def calc_avg_tps(
    input_filename: str,
    prompt_template: str,
    system_prompt: Optional[str],
    model_names: List[str],
    host: str,
    model_type: str,
    **kwargs: Any,
) -> None:
    prompts = await generate_prompts(input_filename, prompt_template)
    for model_name in model_names:
        total_tps = await process_model(
            model_name, prompts, system_prompt, host, model_type, **kwargs
        )
        print(
            f"{model_name},{kwargs['repeat_penalty']},{kwargs['temperature']},{total_tps}"
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
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt for the model",
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
    parser.add_argument("--model_type", type=str, choices=["ollama", "openai"])

    args = parser.parse_args()

    asyncio.run(
        calc_avg_tps(
            input_filename=args.input_filename,
            prompt_template=args.prompt_template,
            system_prompt=args.system_prompt,
            model_names=args.model_names,
            host=args.host,
            model_type=args.model_type,
            temperature=args.temperature,
            repeat_penalty=args.repeat_penalty,
        )
    )


if __name__ == "__main__":
    main()
