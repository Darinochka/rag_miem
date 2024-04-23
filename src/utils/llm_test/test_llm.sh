#!/bin/bash

# Определение хоста и промпт-шаблона
host="http://marisa:11434/api/generate"
prompt_template="Контекст: {context}\\n\\nИспользуя контекст, ответь на вопрос: {query}"

# Перебор файлов
for file in bge-m3 bge-m3-unsupervised; do
    # Перебор моделей
    for model in saiga_mistral_7b_gguf search13b; do
        # Перебор значений температуры
        for temperature in 0.0 0.5 1.0; do
            # Перебор значений repeat_penalty
            for repeat_penalty in 1.0 1.1 1.2; do
                # Формирование имён входных и выходных файлов
                input_filename="data/out_retriever/CS_700_CO_200_q_rerank/${file}.csv"
                output_filename="data/out_llm/CS_700_CO_200_q_rerank/${file}_llm_${model}_temp_${temperature}_rp_${repeat_penalty}_pt1.csv"

                # Запуск скрипта с параметрами
                python3 -m src.utils.llm_test.llm_answer --input_filename "$input_filename" \
                                                --output_filename "$output_filename" \
                                                --prompt_template "$prompt_template" \
                                                --model_name "$model" \
                                                --host "$host" \
                                                --temperature "$temperature" \
                                                --repeat_penalty "$repeat_penalty"
            done
        done
    done
done
