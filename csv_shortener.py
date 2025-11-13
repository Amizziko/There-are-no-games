import pandas as pd
import subprocess
import re

#This program was written with partial assistance from chatgpt

def has_english(lang_str):
    """
    Returns True if 'English' appears in the supported_languages string.
    Handles HTML tags and comma-separated lists.
    """
    if not isinstance(lang_str, str):
        return False
    # Remove HTML tags
    clean = re.sub(r"<.*?>", "", lang_str)
    # Lowercase and remove extra symbols
    clean = clean.lower()
    clean = re.sub(r"[\*\-]", "", clean)  # remove asterisks and dashes
    # Check if 'english' appears anywhere
    return "english" in clean

def main():
    df = pd.read_csv("applications.csv", low_memory=False)

    df = df[df["type"].str.lower() == "game"]

    df = df.dropna(subset=["name", "short_description"])

    df = df[["appid", "name", "short_description"]]

    df["short_description"] = df["short_description"].str.strip()

    subprocess.run(["python3", "reviews_cleaner_upper.py"], check=True)

    reviews = pd.read_csv("different.csv",low_memory=False)
    reviews["voted_up"] = reviews["voted_up"].astype(int)
    avg_reviews = (
        reviews.groupby("appid")["voted_up"]
        .mean()
        .reset_index()
        .rename(columns={"voted_up": "percent_positive"})
    )
    
    merged_list = df.merge(avg_reviews, on="appid", how="left")
    merged_list["percent_positive"] = merged_list["percent_positive"].fillna(50)
    merged_list.to_csv("steam_games_cleaned.csv", index=False)

if __name__ == "__main__":
    main()