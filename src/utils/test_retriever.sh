#!/bin/bash

EMBEDS=("BAAI/bge-m3" "BAAI/bge-m3-unsupervised")
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Итерация по массиву EMBEDS
for EMBED in "${EMBEDS[@]}"; do
    echo "Запускаем обработку для $EMBED c CHUNK_SIZE=$CHUNK_SIZE, CHUNK_OVERLAP=$CHUNK_OVERLAP, RERANKER_MODEL=$RERANKER_MODEL"

    echo "EMBEDDING_MODEL=$EMBED" > .env.override
    echo "CHUNK_SIZE=$CHUNK_SIZE" >> .env.override
    echo "CHUNK_OVERLAP=$CHUNK_OVERLAP" >> .env.override
    echo "RERANKER_MODEL=$RERANKER_MODEL" >> .env.override

    # Запуск Docker Compose с заданной переменной среды
    docker compose up -d retriever

    echo "Засыпаю..."
    sleep 800

    PART=$(echo "$EMBED" | cut -d'/' -f2)
    echo "Выполнение скрипта python"
    python3 src/utils/get_documents.py --input_csv data/validation/test.csv --output_csv data/out_retriever/CS_500_CO_100_q_rerank/"$PART".csv

    echo "Обработка для $EMBED завершена"
done

echo "Все операции завершены."
