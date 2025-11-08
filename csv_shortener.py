import pandas as pd

# Load your dataset
df = pd.read_csv("applications.csv", low_memory=False)

# 1. Keep only games (exclude demos, DLC, software, etc.)
# Make sure the "type" column is lowercase / consistent
df = df[df["type"].str.lower() == "game"]

# Optional: remove rows where name/description are missing
df = df.dropna(subset=["name", "short_description","metacritic_score"])

# 2. Keep only the columns you care about
df = df[["appid", "name", "short_description", "metacritic_score"]]

# 3. (Optional) clean text a little
df["short_description"] = df["short_description"].str.strip()

# 4. Save the cleaned dataset
df.to_csv("steam_games_cleaned.csv", index=False)