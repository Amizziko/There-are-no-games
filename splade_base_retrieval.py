import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
#doc_encodings = torch.load("splade_doc_encodings.pt")
data = torch.load("splade_doc_matrix.pt")

doc_matrix = data["doc_matrix"]
doc_ids = data["doc_ids"]
tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3")
model.to(DEVICE)
model.eval()
vocab_size = 30522


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

    threshold = 0.0008
    new_vector = (splade_vec > threshold).nonzero(as_tuple = True)[1]
    return {int(i): float(splade_vec[0,i]) for i in new_vector}

# def retrieve(query, documents, top_k = 10):
#     scores = {}
#     for doc_id, doc_vec in documents.items():
#         # Sparse dot product
#         score = sum(query.get(k, 0) * v for k, v in doc_vec.items())
#         scores[doc_id] = score
#     # return top k
#     return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

query = "driving a truck in euro"
q_vec = splade_encode(query)
col_idx = torch.tensor(list(q_vec.keys()), dtype=torch.long)
row_idx = torch.zeros(len(col_idx), dtype=torch.long)

query_indices = torch.stack([row_idx, col_idx])

query_values = torch.FloatTensor(list(q_vec.values()))

query_splade = torch.sparse_coo_tensor(query_indices,query_values,(1,vocab_size))
scores = torch.sparse.mm(doc_matrix,query_splade.T).coalesce()
scores = scores.to_dense().squeeze()
top_scores,top_index = torch.topk(scores.squeeze(),k=TOP_K)

df = pd.read_csv("steam_games_cleaned.csv")
for i, score in zip(top_index, top_scores):
    doc_id = doc_ids[i]
    print(f"{doc_id} -> {score.item():.4f}")

    game_name = df.loc[df['appid'] == int(doc_id), "name"].values
    if len(game_name) > 0:
        game_name = game_name[0]
    else:
        game_name = "UNKNOWN"
    print(game_name)
