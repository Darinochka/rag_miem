# MIEM RAG - бот для поиска информации
Бот основан на системе RAG. В этом репозитории хранятся все необходимые скрипты для запуска, конфигурации и тестирования бота.

# Запуск
Чтобы запустить retriever (контактную базу и основную) и telegram-бот, необходимо запустить docker-compose:
```
docker compose up -d
```

Для двух retriever необходимо создать volumes для файлов, которые будут обрабатываться:
```
volumes:
    - ./data/popatruc_wo_qa:/data
    - ~/.cache/huggingface/hub:/.cache/huggingface/hub
    - ./data/faiss_indexes_popatcus_all_wo_qa:/faiss_indexes
```
Здесь ```/data/``` будет хранить файлы для обработки, если необходимо создать новую базу данных. Если такой необходимости нет, можно задать только ```/faiss_indexes``` где будет храниться база данных.

Папка для сканирования файлов связана с ```src/config.toml```. Поэтому при условии изменения папки ```/data```, необходимо поменять значение в ```[documents] folder```.

## Скрипт запуска retriever
В docker-compose запускается скрипт ```src/servers/retriever_server.py```. Его описание следующее:
```
usage: retriever_server.py [-h] [--host HOST] [--port PORT] [--db-load-folder DB_LOAD_FOLDER] [--db-save-folder DB_SAVE_FOLDER] [--bm25 | --no-bm25]

Run the LangChain server

options:
  -h, --help            show this help message and exit
  --host HOST           Host for the FastAPI server
  --port PORT           Port for the FastAPI server
  --db-load-folder DB_LOAD_FOLDER
                        Folder to load FAISS DB from
  --db-save-folder DB_SAVE_FOLDER
                        Folder to save FAISS DB to
  --bm25, --no-bm25     Use BM25 instead of FAISS for retrieval
```
Примеры использования:

Загрузить существующую базу данных из ```/faiss_indexes```:
```
python3 -m src.servers.retriever_server --port 8000 --db-load-folder /faiss_indexes
```

Создать базу данных и сохранить ее в ```/faiss_indexes```:
```
python3 -m src.servers.retriever_server --port 8000 --db-save-folder /faiss_indexes
```

Создать базу данных в контейнере без сохранения:
```
python3 -m src.servers.retriever_server --port 8000
```
# Конфигурация
## Сервис
По умолчанию сервис с основной базой данных будет доступен на порту 8000, а контактная база на порту 8001. Если вы меняете порт, необходимо также поменять эти значения в .env файле.
В example.env можно увидеть пример такого файла:
```
TOKEN=
RETRIEVER_HOST=http://retriever:8000
RETRIEVER_PERSON_HOST=http://retriever-persons:8001
GENERATOR_HOST=
STUDY_CLASSIFIER_HOST=
LLM_NAME=
```
TOKEN - токен от вашего telegram-бота. Его можно получить в боте https://t.me/BotFather

RETRIEVER_HOST - адрес вашего retriever. По умолчанию это http://retriever:8000

RETRIEVER_PERSON_HOST - адрес вашей контактной базы. По умолчанию это http://retriever-persons:8001

GENERATOR_HOST - адрес вашего генератора. Так как в текущей версии используется ollama сервер, здесь должен быть именно он. Например, http://192.168.0.1:11434/api/generate

STUDY_CLASSIFIER_HOST - адрес классификатора, определяющий относится ли запрос к контактной базе или к основной. Необходимо запустить docker-compose в репозитории https://github.com/Darinochka/classifier_bert с моделью на своем комьютере или сервере. На данный момент используется обученный bert. Значение для поля должно выглядеть примерно так: http://192.168.0.1:11434/api/classify

LLM_NAME - название модели, которая используется в генераторе. Например, "llama-2-7b-hf".

## Другое
Для более специфичных настроек вы можете вносить изменения в файл src/config.toml.

В этом файле можно задавать промпт-темплейты, системные промпты, словарь аббривеатур, а также модели для retriever:

### [documents] - настройки для сбора документов
- ```folder``` папка для сканирования документов
- ```target_colum``` колонка в csv файлах, на которых нужно получить эмбеддинги
- ```chunk_size```  длина каждого документа
- ```chunk_overlap```  пересечение документов

### [retriever] - настройки для retriever
- ```embedding_model```  модель для получения эмбеддингов
- ```rerank_model```  модель-реранкер. Если равен None, то использоваться не будет.
- ```normalize_embeddings```  булевая переменная, определяющая, нормализовать ли эмбеддинги
- ```ensemble``` булевая переменная, отвечающая за использование ансамбля моделей. Если равен true, то будет использовать bm25 + embedding_model

### [telegram] - настройки для telegram-бота
- ```study_rooms``` список чисел, которые считаются административным корпусом
- ```system_prompt``` системный промпт для ollama-модели
- ```study_prompt_template``` промпт для ollama-модели, который классифицирует входной промпт на относящийся к учебному процессу или нет
- ```person_prompt_template```  промпт для ollama-модели, который классифицирует входной промпт на относящийся к контактам или нет. По умолчанию такая опция выключена, используется обученный bert. Но достаточно поменять функцию в ```src/servers/telegram_server.py```
- ```prompt_template``` промпт-темплейт для ollama-модели
- ```welcome_text``` текст-приветствие для бота при команде ```/start```
- ```temperature``` - температура для ollama-модели
- ```repeat_penalty``` - штраф для повторения для ollama-модели

### [rewrite_abrr] - словарь для аббривеатур
Здесь необходимо в качестве ключа написать аббривеатуру, а в качестве значения - ее расшифровка.

# Ollama-модель (GENERATOR_HOST)
В качестве модели-generator используется ollama сервер с эндпоинтом /api/generate. Вся информация по запуску здесь https://github.com/ollama/ollama/blob/main/docs/api.md.
