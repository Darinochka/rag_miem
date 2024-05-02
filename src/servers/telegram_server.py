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
from aiogram.enums import ParseMode
import aiogram.types as types
import aiohttp
import json
from typing import Dict, AsyncGenerator

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

dp = Dispatcher()
args = TelegramArgs()
bot = Bot(args.token)
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


def md_autofixer(text: str) -> str:
    # In MarkdownV2, these characters must be escaped: _ * [ ] ( ) ~ ` > # + - = | { } . !
    escape_chars = r"_[]()~>#+-=|{}.!"
    # Use a backslash to escape special characters
    return "".join("\\" + char if char in escape_chars else char for char in text)


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    query = message.text
    logging.info(f"Received query: {query}")

    documents = retrieve_documents(query, args.retriever_host)
    page_content = ""
    for doc in documents["output"]:
        page_content += f"{doc['page_content']}\n\n"
    logging.debug(f"Documents: {page_content}")

    prompt = CONFIG["telegram"]["prompt_template"].format(
        context=page_content, query=query
    )
    await ollama_request(message, prompt, args.generator_host, args.llm_name, bot)


async def generate(
    payload: Dict[str, Any], host: str
) -> AsyncGenerator[Dict[str, Any], Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        async with session.post(host, json=payload) as response:
            async for chunk in response.content:
                if chunk and chunk.decode().strip():
                    yield json.loads(chunk.decode().strip())


async def ollama_request(
    message: types.Message, prompt: str, host: str, model_name: str, bot: Bot
) -> None:
    try:
        await bot.send_chat_action(message.chat.id, "typing")
        full_response = ""
        sent_message = None
        last_sent_text = None
        i = 0

        payload = {
            "model": model_name,
            "stream": True,
            "prompt": prompt,
            "system": CONFIG["telegram"]["system_prompt"],
            "options": {
                "temperature": CONFIG["telegram"]["temperature"],
                "repeat_penalty": CONFIG["telegram"]["repeat_penalty"],
                "num_predict": 2048,
            },
        }
        async for response_data in generate(payload, host):
            if response_data.get("error"):
                logging.error(f"Error from ollama: {response_data['error']}")
                raise Exception(f"{response_data['error']}")

            logging.debug(f"Response data: {response_data}")
            chunk = response_data.get("response")
            if chunk is None:
                continue
            full_response += chunk
            full_response_stripped = full_response.strip()

            # avoid Bad Request: message text is empty
            if full_response_stripped == "":
                continue

            if i == 35 or response_data.get("done"):
                if sent_message:
                    if last_sent_text != full_response_stripped:
                        await bot.edit_message_text(
                            chat_id=message.chat.id,
                            message_id=sent_message.message_id,
                            text=md_autofixer(full_response_stripped),
                            parse_mode=ParseMode.MARKDOWN_V2,
                        )
                        last_sent_text = full_response_stripped
                else:
                    sent_message = await bot.send_message(
                        chat_id=message.chat.id,
                        text=md_autofixer(full_response_stripped),
                        reply_to_message_id=message.message_id,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    last_sent_text = full_response_stripped
                i = 0
            i += 1
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await bot.send_message(
            chat_id=message.chat.id,
            text=md_autofixer(
                f"""Произошла ошибка. Передайте администратору.\n```\n{e}\n```"""
            ),
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def main() -> None:
    await dp.start_polling(bot, skip_update=True)


if __name__ == "__main__":
    asyncio.run(main())
