import asyncio
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.llm_test.llm_answer import summarize_content_ollama
import argparse
import pandas as pd


async def process_questions(args: argparse.Namespace) -> None:
    # Read the input questions from a CSV file
    df = pd.read_csv(args.input_filename)

    # Process each question using the summarize_content_ollama function
    predictions = []
    for _, row in df.iterrows():
        response = await summarize_content_ollama(
            prompt=args.prompt_template.format(question=row["question"]),
            system_prompt=args.system_prompt,
            generator_host=args.host,
            model_name=args.model_name,
            temperature=args.temperature,
            repeat_penalty=args.repeat_penalty,
            num_predict=1,
        )
        predict_class = int(
            response
        )  # Assuming the response can be directly cast to int for simplicity
        predictions.append(predict_class)

    # Add predictions to the dataframe
    df["predict_class"] = predictions

    # Save updated dataframe to the output CSV file
    df.to_csv(args.output_filename, index=False)

    # Calculate metrics
    accuracy = accuracy_score(df["class"], df["predict_class"])
    precision = precision_score(df["class"], df["predict_class"])
    recall = recall_score(df["class"], df["predict_class"])

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process some questions using LLM.")
    parser.add_argument("--host", type=str, required=True, help="Host URL for the LLM")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name for the LLM"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="Temperature setting for the model generation",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        required=False,
        default=1.0,
        help="Repeat penalty setting for the model generation",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=False,
        default=" ",
        help="System prompt for the model generation",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        required=False,
        default=(
            "Тебе нужно классифицировать поисковый запрос на два класса:\n 0 - запрос относится к вузу НИУ ВШЭ, к учебному процессу, факультету "
            "МИЭМ, ПОПАТКУСУ, майнорам, ПУДам, экзаменам, элементам контроля.\n1 - запрос относится к другим темам"
            "Ответь только цифрой 0 или 1.\nВопрос:{question}"
        ),
        help="Prompt template for generation",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        required=True,
        help="Input filename containing the questions and classes in CSV format",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Output filename to save the responses with predicted classes in CSV format",
    )

    args = parser.parse_args()
    asyncio.run(process_questions(args))


if __name__ == "__main__":
    main()
