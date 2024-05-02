import asyncio
import logging

import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from typing import Any
import toml
from src.utils.base_models import TelegramArgs
import re
from aiogram.filters import CommandStart, Command

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

dp = Dispatcher()
args = TelegramArgs()

CONFIG = toml.load("src/config.toml")


def retrieve_documents(query: str, retriever_host: str) -> Any:
    logging.info(f"Starting retrieving documents for query: {query}")
    logging.debug(f"URL for retriever host: {retriever_host}")
    host_url = retriever_host + "/invoke"
    response = requests.post(host_url, json={"input": query})
    res_json = response.json()
    logging.info("Documents were retrieved successfully")
    return res_json


async def summarize_content_ollama(
    context: str, query: str, generator_host: str, model_name: str
) -> Any:
    logging.info("Starting summarization with ollama")
    prompt = CONFIG["telegram"]["prompt_template"].format(context=context, query=query)
    logging.debug(f"Prompt for ollama model: {prompt}")
    response = requests.post(
        generator_host,
        json={
            "model": model_name,
            "stream": False,
            "prompt": prompt,
            "system": CONFIG["telegram"]["system_prompt"],
            "options": {
                "temperature": CONFIG["telegram"]["temperature"],
                "repeat_penalty": CONFIG["telegram"]["repeat_penalty"],
                "num_predict": 2048,
            },
        },
    )
    logging.debug(f"Summary from ollama model: {response.json()['response']}")
    logging.info("Summary was generated successfully")
    return response.json()["response"]


@dp.message(CommandStart())
async def send_welcome(message: Message) -> None:
    welcome_text = CONFIG["telegram"]["welcome_text"]
    await message.answer(welcome_text)


@dp.message(Command("which_building"))
async def which_building(message: Message) -> None:
    text = message.text
    room_number_match = re.search(r"\d+", text)
    if room_number_match:
        room_number = int(room_number_match.group(0))
        if room_number in CONFIG["telegram"]["study_rooms"]:
            response = f"Аудитория {room_number} находится в учебном корпусе."
        elif 0 < room_number < 800:
            response = f"Аудитория {room_number} находится в административном корпусе."
        else:
            response = f"Аудитория {room_number} не найдена."
    else:
        response = "Не удалось распознать номер аудитории. Пожалуйста, укажите команду в формате '/which_building <номер аудитории>'."

    await message.reply(response)


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    query = message.text

    documents = retrieve_documents(query, args.retriever_host)
    page_content = ""
    for doc in documents["output"]:
        page_content += f"{doc['page_content']}\n\n"
    logging.debug(f"Documents: {page_content}")

    summary = await summarize_content_ollama(
        context=page_content,
        query=query,
        generator_host=args.generator_host,
        model_name=args.llm_name,
    )
    if len(summary) > 4096:
        await message.answer(summary[:4096])
    else:
        await message.answer(summary)


async def main() -> None:
    bot = Bot(args.token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
