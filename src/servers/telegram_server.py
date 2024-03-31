import asyncio
import logging

import requests
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from openai import OpenAI

from src.servers.config import (
    GENERATOR_HOST,
    GENERATOR_TYPE,
    MODEL_NAME,
    RETRIEVER_HOST,
    TOKEN,
)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

dp = Dispatcher()

client = OpenAI(
    api_key="dummy",
    base_url=GENERATOR_HOST,
)


def retrieve_documents(query):
    logging.info(f"Начало извлечения документов для запроса: {query}")
    logging.debug(f"URL извлекаемого хоста: {RETRIEVER_HOST}")
    host_url = RETRIEVER_HOST + "/invoke"
    response = requests.post(host_url, json={"input": query})
    res_json = response.json()
    logging.debug(f"Полученный ответ извлечения: {res_json}")
    logging.info("Документы успешно извлечены")
    return res_json


async def summarize_content_openai(input_text, query):
    logging.info("Начало подведения итогов содержания с помощью OpenAI")
    prompt = f"Контекст:\n{input_text}\nВопрос: {query}\nНа основе контекста ответь на вопрос. Не выдумывай, бери ответы ТОЛЬКО из контекста. Ответ:"
    logging.debug(f"Используемый запрос для OpenAI: {prompt}")
    response = client.completion.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=150,
    )
    summary = response.choices[0].text.strip()
    logging.debug(f"Полученный итог от OpenAI: {summary}")
    logging.info("Итоги успешно подведены с помощью OpenAI")
    return summary


async def summarize_content(input_text, query):
    logging.info(
        "Начало подведения итогов содержания с помощью пользовательского генератора"
    )
    prompt = f"Контекст:\n{input_text}\nВопрос:\n{query}\nНа основе контекста ответь на вопрос. Не выдумывай, бери ответы ТОЛЬКО из контекста. Ответ:\n"
    response = requests.post(
        GENERATOR_HOST, json={"model": MODEL_NAME, "stream": False, "prompt": prompt}
    )
    logging.debug(
        f"Полученный ответ от пользовательского генератора: {response.json()}"
    )
    logging.info("Итоги успешно подведены с помощью пользовательского генератора")
    return response.json()["response"]


@dp.message()
async def handle_message(message: Message):
    chat_id = message.from_user.id
    query = message.text
    logging.info(f"Получено сообщение от пользователя {chat_id} с запросом: {query}")
    documents = retrieve_documents(query)
    page_content = "\n\n".join([doc["page_content"] for doc in documents["output"]])
    if GENERATOR_TYPE == "ollama":
        summary = await summarize_content(page_content, query)
    elif GENERATOR_TYPE == "openai":
        summary = await summarize_content_openai(page_content, query)
    else:
        raise ValueError(f"Неизвестный тип генератора: {GENERATOR_TYPE}")
    await message.answer(summary)


async def main() -> None:
    bot = Bot(TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
