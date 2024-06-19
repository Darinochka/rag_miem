#!/bin/bash

EMBEDS=("BAAI/bge-m3")
CHUNK_SIZE=700
CHUNK_OVERLAP=200
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Итерация по массиву EMBEDS
for EMBED in "${EMBEDS[@]}"; do
    echo "Запускаем обработку для $EMBED c CHUNK_SIZE=$CHUNK_SIZE, CHUNK_OVERLAP=$CHUNK_OVERLAP, RERANKER_MODEL=$RERANKER_MODEL"

    PART=$(echo "$EMBED" | cut -d'/' -f2)

    echo "Выполнение скрипта python"

    python3 -m src.utils.retriever_test.retrieve_documents \
        --input_csv data/validation/test.csv \
        --output_csv data/out_retriever/bge-m3_with_bm25.csv \
        --embedding_model $EMBED \
        --chunk_size $CHUNK_SIZE \
        --chunk_overlap $CHUNK_OVERLAP \
        --rerank_model $RERANKER_MODEL \
        --folder data/ready \
        --host http://localhost:8000/invoke

    echo "Обработка для $EMBED завершена"
done

echo "Все операции завершены."
