trafilatura --crawl 1000 --recall -u https://wiki.miem.hse.ru/ | \
grep https://wiki.miem.hse.ru/ | grep -iv https://wiki.miem.hse.ru/Projects/ > data/parser/wiki_links.txt
trafilatura --links -u https://miem.hse.ru/persons  > data/parser/persons_links.txt

trafilatura -i data/parser/wiki_links.txt  --parallel 10 --recall --json > data/wiki_miem.json
trafilatura -i data/parser/hse_links.txt  --parallel 10 --precision --json > data/miem.json
trafilatura -u http://pochemuchnik.hse.ru/ --recall --json > data/precision_pochemuchnik.json

python3 src/parser/parse_persons.py --input_file data/parser/persons_links.txt --output_file data/persons.csv
python3 src/parser/preprocess_persons.py --input-filename data/persons.csv --output-filename data/test/persons_v2.csv

python3 src/parser/json2csv.py --input-filename data/wiki_miem.json --output-filename data/wiki_miem.csv
python3 src/parser/json2csv.py --input-filename data/miem.json --output-filename data/miem.csv
python3 src/parser/json2csv.py --input-filename data/precision_pochemuchnik.json --output-filename data/precision_pochemuchnik.csv

python3 src/parser/preprocess_poch.py --input-filename data/precision_pochemuchnik.csv --output-filename data/pochemuchink.csv
