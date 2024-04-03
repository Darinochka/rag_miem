import typer
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def main(
    peft_model_id: str = typer.Option(default="results"), model_type: str = "t5"
) -> None:
    config = PeftConfig.from_pretrained(peft_model_id)

    if model_type == "gpt":
        # load base LLM model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, load_in_8bit=True
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, load_in_8bit=True
        )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()

    while True:
        try:
            prompt = input("Enter a sentence: ")
            print(prompt)
            temperature = float(input("Enter a temperature: "))
            print(temperature)
            rp = float(input("Enter a repetition penalty: "))
            print(rp)
        except BaseException:
            continue
        if model_type == "t5":
            text = f"<LM>Студент: {prompt}\nАссистент: "
        else:
            text = f"Студент: {prompt}\nАссистент: "
        tokenized = tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        gen = model.generate(
            input_ids=tokenized,
            max_new_tokens=1024,
            temperature=temperature,
            repetition_penalty=rp,
            do_sample=True,
        )
        print(f"Result:\n{tokenizer.decode(gen[0])}")


if __name__ == "__main__":
    typer.run(main)
