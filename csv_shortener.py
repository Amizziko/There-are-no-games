import pandas as pd
import subprocess
import re

def has_english(lang_str):
    """
    Returns True if 'English' appears in the supported_languages string.
    Handles HTML tags and comma-separated lists.
    """
    if not isinstance(lang_str, str):
        return False
    # Remove HTML tags
    clean = re.sub(r"<.*?>", "", lang_str).lower()
    # Split by comma and strip spaces
    langs = [l.strip() for l in clean.split(",")]
    return "english" in langs

def main():
    # Load your dataset
    df = pd.read_csv("applications.csv", low_memory=False)

    # 1. Keep only games (exclude demos, DLC, software, etc.)
    # Make sure the "type" column is lowercase / consistent
    df = df[df["type"].str.lower() == "game"]

    # Optional: remove rows where name/description are missing
    df = df.dropna(subset=["name", "short_description"])

    #df = df[df["supported_languages"].apply(has_english)].copy()

    # 2. Keep only the columns you care about
    df = df[["appid", "name", "short_description"]]

    # 3. (Optional) clean text a little
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
    avg_reviews["percent_positive"] *= 100
    
    merged_list = df.merge(avg_reviews, on="appid", how="inner")
    # 4. Save the cleaned dataset
    #df.to_csv("steam_games_cleaned.csv", index=False)
    merged_list.to_csv("steam_games_cleaned.csv", index=False)

if __name__ == "__main__":
    main()