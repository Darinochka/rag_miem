import argparse
import csv
import json


def main(input_filename, output_filename):
    with open(input_filename, "r") as f, open(output_filename, "w") as fw:
        writer = csv.writer(fw)
        keys = json.loads(f.readline())
        writer.writerow(keys)
        for line in f.readlines():
            data = json.loads(line)
            data["text"] = f'{data["title"]}. {data["text"]}'
            writer.writerow(data.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV.")
    parser.add_argument("--input-filename", help="Input JSON file name")
    parser.add_argument("--output-filename", help="Output CSV file name")
    args = parser.parse_args()

    main(args.input_filename, args.output_filename)
