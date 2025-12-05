import json
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.data.preprocessing import preprocess_data
from src.data.data_loader import split_data_by_group, make_triplets


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_training_examples(triplets: List[dict]) -> List[InputExample]:
    examples: List[InputExample] = []
    for triplet in triplets:
        query = triplet["query"]
        positive = triplet["positive"]
        negatives = triplet["negatives"]
        for neg in negatives:
            examples.append(InputExample(texts=[query, positive, neg]))
    return examples


def load_and_prepare_data(
    data_path: str,
    query_path: str,
    val_size: float,
    test_size: float,
    num_negatives: int,
    random_state: int,
    local_test: bool = False,
    local_rows: int = 2000,
) -> Tuple[pd.DataFrame, List[InputExample]]:
    df = pd.read_csv(data_path)
    df = preprocess_data(df).reset_index(drop=True)

    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Pseudo query file not found: {query_path}")
    queries = pd.read_csv(query_path)
    df = df.merge(queries, left_index=True, right_index=True, how="left")

    if local_test:
        df = df.iloc[:local_rows].copy()

    train_df, val_df, test_df = split_data_by_group(
        df, group_col="title", val_size=val_size, test_size=test_size, random_state=random_state
    )

    triplets = make_triplets(train_df, num_negatives=num_negatives, random_state=random_state)
    train_examples = build_training_examples(triplets)

    return df, train_examples


def train_biencoder(config_path: str, local_test: bool = False):
    config = load_json(config_path)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    device = config.get("device", "cpu")
    random_state = config.get("random_state", 42)

    set_seed(random_state)

    data_path = data_cfg.get("path", "data/winemag-data-130k-v2.csv")
    query_path = data_cfg.get("query_path", "data/pseudo_queries.csv")
    val_size = data_cfg.get("val_size", 0.1)
    test_size = data_cfg.get("test_size", 0.1)
    num_negatives = data_cfg.get("num_negatives", 2)

    model_name = model_cfg.get("model_name", "all-mpnet-base-v2")
    train_batch_size = train_cfg.get("train_batch_size", model_cfg.get("batch_size", 32))
    triplet_margin = train_cfg.get("triplet_margin", 0.5)

    epochs = train_cfg.get("epochs", 3)
    learning_rate = train_cfg.get("learning_rate", 2e-5)
    warmup_steps = train_cfg.get("warmup_steps")
    grad_accum = train_cfg.get("gradient_accumulation", 1)
    use_amp = train_cfg.get("use_amp", True)
    output_dir = train_cfg.get("output_dir", "models/bi-encoder-trained")

    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"

    full_df, train_examples = load_and_prepare_data(
        data_path=data_path,
        query_path=query_path,
        val_size=val_size,
        test_size=test_size,
        num_negatives=num_negatives,
        random_state=random_state,
        local_test=local_test,
    )

    model = SentenceTransformer(model_name, device=device)
    train_dataloader = DataLoader(train_examples, batch_size=train_batch_size, shuffle=True)
    train_loss = losses.TripletLoss(
        model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=triplet_margin
    )

    if warmup_steps is None:
        warmup_steps = int(len(train_dataloader) * 0.1)

    print(
        f"\n[Train] bi-encoder start | epochs={epochs}, lr={learning_rate}, "
        f"bs={train_batch_size}, margin={triplet_margin}, grad_accum={grad_accum}"
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        use_amp=use_amp,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
        gradient_accumulation_steps=grad_accum,
    )

    print(f"\n[Train] Finished. Model saved to: {output_dir}")


__all__ = ["train_biencoder"]
