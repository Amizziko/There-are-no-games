import pandas as pd
from sentence_transformers import InputExample
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
df = pd.read_csv("steam_games_cleaned.csv")
df = df.dropna(subset=["appid","name","short_description"])


# Create InputExamples
train_examples = [
    InputExample(texts=[row["short_description"], row["appid"]],
                 label=1.0)
    for _, row in df.iterrows()
]


##part below not done pls ignore
# Weighted sampler for DataLoader
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def collate_fn(batch):
    return batch
train_dataloader = DataLoader(train_examples, batch_size=16, sampler=sampler)

