
#This function was written by chatgpt but this is only a helper file to merge all encodings togetther to preserve memory
#Changed to making a matrix to save time when computing matching entries
# Open a persistent dict on disk
import torch
import glob
import os
from tqdm import tqdm


files = glob.glob("splade_doc_encodings_*.pt")
indices = []
values = []
document_ids = []
document_idx = 0
vocab_size = 30522


for f in tqdm(files, desc="Merging chunk files"):
        print(f"Processing file: {f}")
        part = torch.load(f)
        for doc_id, vec in part.items():
            document_ids.append(doc_id)
            for tokenindex, weight in vec.items():
                indices.append([document_idx,tokenindex])
                values.append(weight)
            document_idx += 1

documents_number = len(indices)

indices = torch.LongTensor(indices).T
values = torch.FloatTensor(values)

matrix = torch.sparse_coo_tensor(indices,values,size=(documents_number, vocab_size))
torch.save({
    "doc_matrix": matrix,
    "doc_ids": document_ids
}, "splade_doc_matrix.pt")

if os.path.exists("splade_doc_matrix.pt"):
    for f in glob.glob("splade_doc_encodings_*.pt"):
        os.remove(f)
        print(f"Deleted {f}")
else:
    print("Final matrix not found — not deleting chunks")