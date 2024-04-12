import asyncio
import logging

import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from openai import OpenAI
from typing import Any

from src.utils.base_models import TelegramArgs

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

dp = Dispatcher()
args = TelegramArgs()

client = OpenAI(
    api_key="dummy",
    base_url=args.generator_host,
)

PROMPT_TEMPLATE = (
    "На основе контекста ответь на вопрос. Не выдумывай, бери ответы ТОЛЬКО из контекста. "
    # "Если в контексте нет ответа, ответь 'К сожалению, нет информации связанной с этим вопросом.'\n"
    "Контекст:\n{context}\n"
    "Вопрос:\n{query}\n"
    "Ответ:\n"
)


def retrieve_documents(query: str, retriever_host: str) -> Any:
    logging.info(f"Starting retrieving documents for query: {query}")
    logging.debug(f"URL for retriever host: {retriever_host}")
    host_url = retriever_host + "/invoke"
    response = requests.post(host_url, json={"input": query})
    res_json = response.json()
    logging.info("Documents were retrieved successfully")
    return res_json


async def summarize_content_openai(
    context: str, query: str, client: OpenAI, model_name: str
) -> Any:
    logging.info("Starting summarization with OpenAI")
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    logging.debug(f"Prompt for OpenAI model: {prompt}")
    response = client.completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=150,
    )
    summary = response.choices[0].text.strip()
    logging.debug(f"Summary from OpenAI model: {summary}")
    logging.info("Summary was generated successfully")
    return summary


async def summarize_content_ollama(
    context: str, query: str, generator_host: str, model_name: str
) -> Any:
    logging.info("Starting summarization with ollama")
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    logging.debug(f"Prompt for ollama model: {prompt}")
    response = requests.post(
        generator_host, json={"model": model_name, "stream": False, "prompt": prompt}
    )
    logging.debug(f"Summary from ollama model: {response.json()['response']}")
    logging.info("Summary was generated successfully")
    return response.json()["response"]


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    query = message.text

    documents = retrieve_documents(query, args.retriever_host)
    page_content = "\n\n".join([doc["page_content"] for doc in documents["output"]])
    logging.debug(f"Documents: {page_content}")

    if args.generator_type == "ollama":
        summary = await summarize_content_ollama(
            context=page_content,
            query=query,
            generator_host=args.generator_host,
            model_name=args.llm_name,
        )
    elif args.generator_type == "openai":
        summary = await summarize_content_openai(
            context=page_content,
            query=query,
            client=client,
            model_name=args.llm_name,
        )
    else:
        raise ValueError(f"Unknown the type of the generator: {args.generator_type}")
    await message.answer(summary)


async def main() -> None:
    bot = Bot(args.token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
