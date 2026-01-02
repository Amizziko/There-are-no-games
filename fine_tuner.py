from fine_tune_generate_test_train_val_query import prepare_training
import pandas as pd
import torch
from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer,  SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset as HFDataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SparseEncoder("naver/splade-v3")
model.to(DEVICE)
model.eval()
def main():
    training_queries, training_documents, validation_queries, validation_documents, _,_ = prepare_training()

    training_dataset = HFDataset.from_dict({"query": training_queries, "document": training_documents})
    validation_dataset = HFDataset.from_dict({"query": validation_queries, "document": validation_documents})
    
    
    #Source: https://huggingface.co/blog/train-sparse-encoder#trainer
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
    model.save_pretrained(f"models/There-Are-No-Games")

    #model.push_to_hub("mazombieme/There-Are-No-Games") #Comment out otherwise you WILL get an error
                                    

if __name__ == "__main__":
    main()