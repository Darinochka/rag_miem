#!/bin/bash

host="http://marisa:11434/api/generate"
prompt_template="Контекст: {context}\\n\\nИспользуя контекст, ответь на вопрос: {query}"
file=bge-m3
input_filename="data/out_retriever/CS_700_CO_200_q_rerank/${file}.csv"

python3 -m src.utils.llm_test.tps_benchmark --input_filename "$input_filename" \
                                --prompt_template "$prompt_template" \
                                --model_names saiga-mistral-7b-q4:latest \
                                --host "$host" \
                                --temperature 0.0 \
                                --repeat_penalty 1.1
