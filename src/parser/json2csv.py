import argparse
import csv
import json


def main(input_filename: str, output_filename: str) -> None:
    with open(input_filename, "r") as f, open(output_filename, "w") as fw:
        writer = csv.writer(fw)
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            if i == 0:
                writer.writerow(data)
            data["text"] = f'{data["title"]}. {data["text"]}'
            writer.writerow(data.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV.")
    parser.add_argument("--input-filename", type=str, help="Input JSON file name")
    parser.add_argument("--output-filename", type=str, help="Output CSV file name")
    args = parser.parse_args()

    main(args.input_filename, args.output_filename)
