import pandas as pd
from sentence_transformers import InputExample

df = pd.read_csv("steam_games_short.csv")
df = df.dropna(subset=["short_description", "name", "metacritic_score"])

# Normalize metacritic_score to [0,1]
min_score = df["metacritic_score"].min()
max_score = df["metacritic_score"].max()
df["weight"] = (df["metacritic_score"] - min_score) / (max_score - min_score)
