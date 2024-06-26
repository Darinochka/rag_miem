import argparse
import pandas as pd


def cut_length(text: str, max_len: int) -> str:
    if isinstance(text, str) and len(text) > max_len:
        return text[:max_len]
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename", type=str, required=True)
    parser.add_argument("--output-filename", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=700)
    args = parser.parse_args()

    data = pd.read_csv(args.input_filename)
    data["text"] = data["text"].apply(cut_length, args=(args.max_len,))
    data = data.dropna(subset=["text"])
    data = data.drop_duplicates(subset=["source"], keep="last")
    data["text"] = (
        data["text"]
        .apply(
            str.replace,
            "Московский институт электроники и математики им. А.Н. Тихонова",
            "",
        )
        .str.strip()
    )
    data.to_csv(args.output_filename, index=False)


if __name__ == "__main__":
    main()
