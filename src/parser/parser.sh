trafilatura --crawl 1000 --recall -u https://wiki.miem.hse.ru/ | grep https://wiki.miem.hse.ru/ | grep -iv https://wiki.miem.hse.ru/Projects/ > data/parser/wiki_links.txt

trafilatura -i data/parser/wiki_links.txt  --parallel 10 --precision --json > data/wiki_miem.json
trafilatura -i data/parser/hse_links.txt  --parallel 10 --precision --json > data/miem.json

python3 src/parser/json2csv.py --input-filename data/wiki_miem.json --output-filename data/wiki_miem.csv
python3 src/parser/json2csv.py --input-filename data/miem.json --output-filename data/miem.csv
