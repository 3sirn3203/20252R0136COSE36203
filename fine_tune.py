import os
import pandas as pd
import torch
import warnings
import argparse
import json
import random
import math
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

from src.data.downloader import download_dataset
from src.data.preprocessing import preprocess_data


load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')

DATA_PATH = "data/winemag-data-130k-v2.csv"
CONFIG_PATH = "config/fine_tune_mlm.json"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(file_path: str):
    """JSON 파일을 읽어 딕셔너리로 반환하는 함수"""
    print(f"Loading JSON config from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_data(df: pd.DataFrame, val_size: float, random_state: int):

    df = df[df['combined_text'].str.len() > 20]
    combined_texts = df['combined_text'].tolist()

    print(f"Total samples: {len(combined_texts)}")

    train_texts, val_texts = train_test_split(
        combined_texts,
        test_size=val_size,
        random_state=random_state
    )

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return train_dataset, val_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="all-mpnet-base-v2 모델을 MLM 방식으로 파인튜닝")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="config 파일 경로")
    args = parser.parse_args()

    config_path = args.config
    config = load_json(config_path)

    data_config = config.get("data", {})
    data_path = data_config.get("path", DATA_PATH)
    val_size = data_config.get("val_size", 0.1)

    model_config = config.get("model", {})
    model_name = model_config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
    max_seq_length = model_config.get("max_seq_length", 256)
    batch_size = model_config.get("batch_size", 16)
    epochs = model_config.get("epochs", 3)
    learning_rate = model_config.get("learning_rate", 5e-5)
    mlm_probability = model_config.get("mlm_probability", 0.15)
    weight_decay = model_config.get("weight_decay", 0.01)
    eval_save_strategy = model_config.get("eval_save_strategy", "epoch")
    logging_steps = model_config.get("logging_steps", 100)

    random_state = data_config.get("random_state", 42)
    output_dir = model_config.get("output_dir", "models/fine_tuned_mlm")

    set_seed(random_state)

    if not os.path.exists(data_path):
        download_dataset()

    df = pd.read_csv(data_path)

    df_preprocessed = preprocess_data(df)
    df_preprocessed = df_preprocessed.reset_index(drop=True)

    train_dataset, val_dataset = prepare_data(
        df=df_preprocessed,
        val_size=val_size,
        random_state=random_state
    )

    print(f"Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length
        )
    print(f"Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_strategy=eval_save_strategy,
        save_strategy=eval_save_strategy,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    print("\n>>> Starting MLM Fine-tuning...")
    train_result = trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    print(f"\nSaving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Done! You can now load this model with SentenceTransformer.")