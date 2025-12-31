from fine_tune_generate_test_train_val_query import prepare_training
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer,  SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset as HFDataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
#model = AutoModelForSequenceClassification.from_pretrained("naver/splade-v3")
model = SparseEncoder("naver/splade-v3")
model.to(DEVICE)
model.eval()

class Spladedataset(Dataset):
    def __init__(self,queries, documents):
        self.queries = queries
        self.documents = documents
        self.column_names = ["query", "document"]
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, i):

        #q_enc = self.tokenizer(q, truncation=True, max_length=self.max_len, return_tensors="pt")
        #d_enc = self.tokenizer(d, truncation=True, max_length=self.max_len, return_tensors="pt")

        return {"query": self.queries[i], "document": self.documents[i]}
    
def main():
    training_queries, training_documents, validation_queries, validation_documents, our_test_queries,test_documents = prepare_training()
    #training_dataset = Spladedataset(training_queries,training_documents)
    #validation_dataset = Spladedataset(validation_queries,validation_documents)
    

    training_dataset = HFDataset.from_dict({"query": training_queries, "document": training_documents})
    validation_dataset = HFDataset.from_dict({"query": validation_queries, "document": validation_documents})
    #Source: https://huggingface.co/blog/train-sparse-encoder#trainer
    # 3. Load a dataset to finetune on
    #full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    #dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    
    train_dataset = training_dataset
    eval_dataset = validation_dataset
    print(train_dataset)
    print(train_dataset[0])

    # 4. Define a loss function
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=0,
        document_regularizer_weight=3e-3,
    )

    # 5. (Optional) Specify training arguments
    run_name = "inference-free-splade-distilbert-base-uncased-nq"
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        router_mapping={"query": "query", "answer": "document"},  # Map the column names to the routes
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )


    # 7. Create a trainer & train
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss#,
        #evaluator=dev_evaluator,
    )
    trainer.train()

    # 9. Save the trained model
    model.save_pretrained(f"models/{run_name}/final")

    #model.push_to_hub("mazombieme/There-Are-No-Games") #Comment out otherwise you WILL get an error
                                    

if __name__ == "__main__":
    main()