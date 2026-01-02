from fine_tune_generate_test_train_val_query import prepare_training
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder import SparseEncoderModelCardData


"""
    Helper function for turning list of queries and documents into a
    corpus usable by: InformationRetrievalEvaluator
    Code adapted from a chatgpt prompt. 
    """
def generate_ir_eval_corpus(queries, documents):
    
    corpus = {}
    query_dict = {}
    relevant_docs = {}
    for i, (query, doc) in enumerate(zip(queries, documents)):
        q_id = f"query{i}"
        doc_id = f"document{i}"
        query_dict[q_id] = query
        corpus[doc_id] = doc
        relevant_docs[q_id] = {doc_id}
    return query_dict, corpus, relevant_docs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SparseEncoder("mazombieme/There-Are-No-Games")
if not hasattr(model, "model_card_data") or model.model_card_data is None:
    model.model_card_data = SparseEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Custom Finetuned SPLADEv3 on Steam dataset"
    )
default_model = SparseEncoder("naver/splade-v3")
model.to(DEVICE)
default_model.to(DEVICE)
model.eval()
default_model.eval()

_,_,_,_, our_test_queries,test_documents = prepare_training()
query_dict, corpus, relevant_test_docs = generate_ir_eval_corpus(our_test_queries,test_documents)
# Adapted from Source: https://github.com/huggingface/sentence-transformers/blob/main/examples/sparse_encoder/evaluation/sparse_retrieval_evaluator.py
evaluator = SparseInformationRetrievalEvaluator(
        queries=query_dict,
        corpus=corpus,
        relevant_docs=relevant_test_docs,
        name="Steam-dataset-subset-test",
        show_progress_bar=True,
        batch_size=16,
    )



print("DEFAULT SPLADEV3")
results = evaluator(default_model)
# === DEFAULT SPLADEV3 ===
# accuracy@1 : 0.7785
# accuracy@3 : 0.8525
# accuracy@5 : 0.8832
# accuracy@10 : 0.9130
# precision@1 : 0.7785
# precision@3 : 0.2842
# precision@5 : 0.1766
# precision@10 : 0.0913
# recall@1 : 0.7785
# recall@3 : 0.8525
# recall@5 : 0.8832
# recall@10 : 0.9130
# ndcg@10 : 0.8445
# mrr@10 : 0.8227
# map@100 : 0.8257
# query_active_dims : 27.4402
# query_sparsity_ratio : 0.9991
# corpus_active_dims : 191.8756
# corpus_sparsity_ratio : 0.9937
# _avg_flops : 4.8365


#print(results.items())
# => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10




custom_results = evaluator(model)
# === Custom Finetuned Splade v3 ===
# accuracy@1 : 0.8917
# accuracy@3 : 0.9592
# accuracy@5 : 0.9748
# accuracy@10 : 0.9888
# precision@1 : 0.8917
# precision@3 : 0.3197
# precision@5 : 0.1950
# precision@10 : 0.0989
# recall@1 : 0.8917
# recall@3 : 0.9592
# recall@5 : 0.9748
# recall@10 : 0.9888
# ndcg@10 : 0.9435
# mrr@10 : 0.9286
# map@100 : 0.9292
# query_active_dims : 107.8732
# query_sparsity_ratio : 0.9965
# corpus_active_dims : 98.3774
# corpus_sparsity_ratio : 0.9968
# avg_flops : 1.6232


#Helper function written by chatgpt
def print_results(title, results_dict):
    print(f"\n=== {title} ===")
    for metric, value in results_dict.items():
        print(f"{metric:30} : {value:.4f}")


print_results("DEFAULT SPLADEV3", results)
print_results("Custom Finetuned Splade v3", custom_results)
evaluator.store_metrics_in_model_card_data(model,custom_results)

model.save_pretrained("models/There-Are-No-Games")
model.push_to_hub("mazombieme/There-Are-No-Games", commit_message="Update model card with evaluation metrics", exist_ok=True)