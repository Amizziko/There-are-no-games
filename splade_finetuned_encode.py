import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_AFTER = 5000

tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
model = AutoModelForMaskedLM.from_pretrained("mazombieme/There-Are-No-Games")
model.to(DEVICE)
model.eval()

#Note i split it into seperate files, because otherwise my PC runs out of ram and this would crash
@torch.no_grad()
def splade_encode(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**inputs)
    logits = outputs.logits
    weights = torch.log1p(torch.relu(logits))
    splade_vec = torch.max(weights, dim=1).values

    return splade_vec.cpu()

def convert_to_sparse_vec(vector, threshold = 0.0008):
    new_vector = (vector > threshold).nonzero(as_tuple = True)[0]
    return {int(i): float(vector[i]) for i in new_vector}

df = pd.read_csv("steam_games_cleaned.csv")

encoded_games = {}
batch_size = 32
batch_texts = []
app_ids = []
counter = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    name = row["name"]
    description = row["short_description"]
    text = f"{name}. {description}"
    document_id = str(row["appid"])

    batch_texts.append(text)
    app_ids.append(document_id)

    if len(batch_texts) == batch_size:
        vectors = splade_encode(batch_texts)
        for id, vector in zip(app_ids, vectors):
            encoded_games[id] = convert_to_sparse_vec(vector)
            counter += 1
            if counter % SAVE_AFTER == 0:
                torch.save(encoded_games, f"custom_doc_encodings_{counter}.pt")
                encoded_games = {}
        batch_texts, app_ids = [],[]

if batch_texts:
    vectors = splade_encode(batch_texts)
    for id, vector in zip(app_ids, vectors):
        encoded_games[id] = convert_to_sparse_vec(vector)
        counter += 1
        if counter % SAVE_AFTER == 0:
                torch.save(encoded_games, f"custom_doc_encodings_{counter}.pt")
                encoded_games = {}

torch.save(encoded_games, "custom_doc_encodings_rest.pt")