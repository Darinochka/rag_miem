#!/bin/bash

host="http://marisa:11434/api/generate"
prompt_template="Контекст: {context}\\n\\nИспользуя контекст, ответь на вопрос: {query}"
yandex_token=""
folder_id=""

for file in bge-m3; do
    for model in saiga2-7b-q4:latest; do
        for temperature in 0.0 0.5 1.0; do
            for repeat_penalty in 1.0 1.1 1.2; do
                input_filename="data/out_retriever/CS_700_CO_200_q_rerank/${file}.csv"
                output_filename="data/out_llm/CS_700_CO_200_q_rerank/${file}_llm_${model}_temp_${temperature}_rp_${repeat_penalty}_pt1.csv"

                python3 -m src.utils.llm_test.llm_answer --input_filename "$input_filename" \
                                                --output_filename "$output_filename" \
                                                --prompt_template "$prompt_template" \
                                                --model_name "$model" \
                                                --host "$host" \
                                                --temperature "$temperature" \
                                                --repeat_penalty "$repeat_penalty" \
                                                --token "$yandex_token" \
                                                --folder_id "$folder_id" \
                                                --model_type ollama
                sleep 100
            done
        done
    done
done
