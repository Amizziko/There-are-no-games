import csv
import sys

#
#This file was written by copilot
#
#


#!/usr/bin/env python3

IN_FILE = "reviews.csv"
OUT_FILE = "different.csv"

REQUIRED_COLS = [
    "recommendationid",
    "appid",
    "author_num_reviews",
    "author_playtime_forever",
    "language",
    "review_text",
    "voted_up",  # take the first occurrence of this column name
]

def find_first_index(header_lower, name):
    for i, h in enumerate(header_lower):
        if h == name:
            return i
    return None

def to_int_or_none(s):
    try:
        return int(float(s))
    except Exception:
        return None

def main(in_file=IN_FILE, out_file=OUT_FILE):
    with open(in_file, newline='', encoding='utf-8') as inf, \
         open(out_file, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)

        try:
            header = next(reader)
        except StopIteration:
            return

        header_lower = [h.strip().lower() for h in header]

        # find indices for required columns (use first occurrence)
        idx_map = {}
        for name in REQUIRED_COLS:
            idx = find_first_index(header_lower, name)
            if idx is None:
                # missing required column -> stop
                raise SystemExit(f"Missing required column in input CSV: {name}")
            idx_map[name] = idx

        # write output header (use canonical names)
        writer.writerow(REQUIRED_COLS)

        for row in reader:
            # guard against short rows
            if len(row) < max(idx_map.values()) + 1:
                continue

            # extract values
            rec_id = row[idx_map["recommendationid"]].strip()
            appid = row[idx_map["appid"]].strip()
            num_reviews_s = row[idx_map["author_num_reviews"]].strip()
            playtime_s = row[idx_map["author_playtime_forever"]].strip()
            language = row[idx_map["language"]].strip().lower()
            review_text = row[idx_map["review_text"]].strip()
            voted_up = row[idx_map["voted_up"]].strip()

            # filters
            num_reviews = to_int_or_none(num_reviews_s)
            playtime = to_int_or_none(playtime_s)
            if num_reviews is None or playtime is None:
                continue
            if num_reviews < 4 or playtime < 4:
                continue
            if language != "english":
                continue

            writer.writerow([rec_id, appid, str(num_reviews), str(playtime), "english",review_text, voted_up])

if __name__ == "__main__":
    # optional: allow passing filenames as args
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()