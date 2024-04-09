import asyncio
import logging

import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from openai import OpenAI
from typing import Any

from src.servers.base_models import TelegramArgs

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

dp = Dispatcher()
args = TelegramArgs()

client = OpenAI(
    api_key="dummy",
    base_url=args.generator_host,
)


def retrieve_documents(query: str, retriever_host: str) -> Any:
    logging.info(f"Начало извлечения документов для запроса: {query}")
    logging.debug(f"URL извлекаемого хоста: {retriever_host}")
    host_url = retriever_host + "/invoke"
    response = requests.post(host_url, json={"input": query})
    res_json = response.json()
    logging.debug(f"Полученный ответ извлечения: {res_json}")
    logging.info("Документы успешно извлечены")
    return res_json


async def summarize_content_openai(
    context: str, query: str, client: OpenAI, model_name: str
) -> Any:
    logging.info("Начало подведения итогов содержания с помощью OpenAI")
    prompt = f"Контекст:\n{context}\nВопрос: {query}\nНа основе контекста ответь на вопрос. Не выдумывай, бери ответы ТОЛЬКО из контекста. Ответ:"
    logging.debug(f"Используемый запрос для OpenAI: {prompt}")
    response = client.completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=150,
    )
    summary = response.choices[0].text.strip()
    logging.debug(f"Полученный итог от OpenAI: {summary}")
    logging.info("Итоги успешно подведены с помощью OpenAI")
    return summary


async def summarize_content_ollama(
    context: str, query: str, generator_host: str, model_name: str
) -> Any:
    logging.info(
        "Начало подведения итогов содержания с помощью пользовательского генератора"
    )
    prompt = f"Контекст:\n{context}\nВопрос:\n{query}\nНа основе контекста ответь на вопрос. Не выдумывай, бери ответы ТОЛЬКО из контекста. Ответ:\n"
    response = requests.post(
        generator_host, json={"model": model_name, "stream": False, "prompt": prompt}
    )
    logging.debug(
        f"Полученный ответ от пользовательского генератора: {response.json()}"
    )
    logging.info("Итоги успешно подведены с помощью пользовательского генератора")
    return response.json()["response"]


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    chat_id = message.from_user.id
    query = message.text
    logging.info(f"Получено сообщение от пользователя {chat_id} с запросом: {query}")
    documents = retrieve_documents(query, args.retriever_host)
    page_content = "\n\n".join([doc["page_content"] for doc in documents["output"]])
    if args.generator_type == "ollama":
        summary = await summarize_content_ollama(
            context=page_content,
            query=query,
            generator_host=args.generator_host,
            model_name=args.llm,
        )
    elif args.generator_type == "openai":
        summary = await summarize_content_openai(
            context=page_content,
            query=query,
            client=client,
            model_name=args.llm,
        )
    else:
        raise ValueError(f"Неизвестный тип генератора: {args.generator_type}")
    await message.answer(summary)


async def main() -> None:
    bot = Bot(args.token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
