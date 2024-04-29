#!/bin/bash

EMBEDS=("BAAI/bge-m3" "BAAI/bge-m3-unsupervised")
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Итерация по массиву EMBEDS
for EMBED in "${EMBEDS[@]}"; do
    echo "Запускаем обработку для $EMBED c CHUNK_SIZE=$CHUNK_SIZE, CHUNK_OVERLAP=$CHUNK_OVERLAP, RERANKER_MODEL=$RERANKER_MODEL"

    PART=$(echo "$EMBED" | cut -d'/' -f2)

    echo "Выполнение скрипта python"

    python3 -m src.utils.retriever_test.retriever_documents \
        --input_csv data/validation/test.csv \
        --output_csv data/out_retriever/CS_"$CHUNK_SIZE"_CO_"$CHUNK_OVERLAP"_q_rerank/"$PART".csv \
        --embedding_model $EMBED \
        --chunk_size $CHUNK_SIZE \
        --chunk_overlap $CHUNK_OVERLAP \
        --rerank_model $RERANKER_MODEL \
        --folder data/ready

    echo "Обработка для $EMBED завершена"
done

echo "Все операции завершены."
