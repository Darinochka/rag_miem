import logging
import os
from typing import Any, Dict, Hashable
from uuid import uuid4

import typer
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

os.environ["WANDB_PROJECT"] = "miem-llm"


def get_data(filename: str) -> Dict[Hashable, Dict[int, str]]:
    logging.info(f"Getting dataset with name {filename}")

    dataset = load_dataset("csv", data_files=filename)

    logging.info(f"Columns of the dataset {dataset['train'].features}")

    return dataset


def preprocess_for_alpaca(
    example, tokenizer, prefix: str = "<LM>", max_length: int = 512
):
    question = f"Студент: {example['instruction']} {example['input']}\nАссистент: "
    answer = example["output"] + "</s>"

    return tokenize_t5(tokenizer, question, answer, max_length=max_length)


def preprocess_for_miem(
    example, tokenizer, preifx: str = "<LM>", max_length: int = 512
):
    question = f"Студент: {example['question']}\nАссистент: "
    answer = example["answer"] + "</s>"

    return tokenize_t5(tokenizer, question, answer, max_length=max_length)


def tokenize_t5(tokenizer, question, answer, max_length: int):
    tokenized = tokenizer(
        question,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    labels = tokenizer(
        answer,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    tokenized["labels"] = labels["input_ids"].flatten()
    tokenized["input_ids"] = tokenized["input_ids"].flatten()
    tokenized["attention_mask"] = tokenized["attention_mask"].flatten()

    return tokenized


def main(
    dataset_filename: str = typer.Option(),
    model_name: str = typer.Option(default="ai-forever/FRED-T5-large"),
    output_dir: str = typer.Option(default="checkpoints/test1"),
    epoch: int = typer.Option(default=5),
    lora: bool = typer.Option(default=True),
    context: int = typer.Option(default=512),
    lr: float = typer.Option(default=1e-2),
    lora_r: int = typer.Option(default=16),
    lora_alpha: int = typer.Option(default=16),
    lora_dropout: float = typer.Option(default=0.05),
    prefix: str = typer.Option(default=""),
):
    model = T5ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=lora)
    tokenizer = AutoTokenizer.from_pretrained(model_name, eos_token_id=2)

    print(model)
    dataset = get_data(filename=dataset_filename)
    dataset["train"] = dataset["train"].map(
        preprocess_for_miem, fn_kwargs={"tokenizer": tokenizer, "max_length": context}
    )

    alpaca = load_dataset("IlyaGusev/ru_turbo_alpaca", split="train[:5%]")
    alpaca = alpaca.map(
        preprocess_for_alpaca, fn_kwargs={"tokenizer": tokenizer, "max_length": context}
    )

    dataset["train"] = concatenate_datasets([dataset["train"], alpaca])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(dataset["train"][0])
    print(dataset["train"][-1])

    uuid_gen = str(uuid4()).split("-")[0]
    task_name = f"{model_name}_lr{lr}_epoch{epoch}_lora{lora}_lorar_{lora_r}_loraa_{lora_alpha}_{context}_{uuid_gen}"
    output_dir = "checkpoints/" + task_name

    if lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # auto_find_batch_size=True,
        learning_rate=lr,
        num_train_epochs=epoch,
        logging_strategy="steps",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        report_to=["wandb"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    os.environ["WANDB_NAME"] = task_name

    trainer.train()

    peft_model_id = "results"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)


if __name__ == "__main__":
    typer.run(main)
