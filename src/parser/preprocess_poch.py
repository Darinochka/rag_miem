import pandas as pd
import argparse


def remove_consecutive_duplicates_in_line(text: str) -> str:
    elements = text.split("\n")
    unique_elements = [
        elements[i]
        for i in range(len(elements))
        if i == 0 or elements[i] != elements[i - 1]
    ]
    return "\n".join(unique_elements)


def process_csv(input_filename: str, output_filename: str) -> None:
    df = pd.read_csv(input_filename)

    df["text"] = df["text"].apply(remove_consecutive_duplicates_in_line)
    df.to_csv(output_filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove consecutive duplicate lines from a CSV file."
    )
    parser.add_argument(
        "--input-filename", type=str, required=True, help="Input CSV filename"
    )
    parser.add_argument(
        "--output-filename", type=str, required=True, help="Output CSV filename"
    )

    args = parser.parse_args()

    process_csv(args.input_filename, args.output_filename)


if __name__ == "__main__":
    main()
