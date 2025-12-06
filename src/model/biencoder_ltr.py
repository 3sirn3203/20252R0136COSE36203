import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(model_config: dict,
                 train_config: dict,
                 train_triplets, 
                 val_df, 
                 device: str, 
                 random_state: int) -> SentenceTransformer:
    
    """Config 설정을 받아 모델 생성"""
    set_seed(random_state)
    
    model_name = model_config.get("model_name", "all-mpnet-base-v2")
    max_seq_length = model_config.get("max_seq_length", 256)

    re_train = train_config.get("re_train", True)
    epochs = train_config.get("epochs", 1)
    learning_rate = train_config.get("learning_rate", 2e-5)
    scheduler = train_config.get("scheduler", "WarmupLinear")
    warmup_steps = train_config.get("warmup_steps", 1000)
    train_batch_size = train_config.get("train_batch_size", 32)
    weight_decay = train_config.get("weight_decay", 0.01)
    max_grad_norm = train_config.get("max_grad_norm", 1.0)
    use_amp = train_config.get("use_amp", True)
    loss_function = train_config.get("loss_function", "TripletLoss")
    triplet_margin = train_config.get("triplet_margin", 0.5)
    output_dir = train_config.get("output_dir", "models/biencoder_ltr_trained")
    eval_steps = train_config.get("eval_steps", 500)
    save_best_model = train_config.get("save_best_model", True)

    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"

    if not re_train and output_dir and os.path.exists(output_dir):
        print(f"\nLoading existing model from {output_dir} (re_train=False).")
        model = SentenceTransformer(output_dir, device=device)
        model.max_seq_length = max_seq_length
        return model

    print(f"\nCreating biencoder model '{model_name}' on device '{device}'...")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    print(f"Preparing training with {len(train_triplets)} triplets...")
    train_examples = []
    for triplet in train_triplets:
        query = triplet['query']
        pos = triplet['positive']
        for neg in triplet['negatives']:
            train_examples.append(InputExample(texts=[query, pos, neg]))

    train_dataloader = DataLoader(train_examples, batch_size=train_batch_size, shuffle=True)

    # 손실 함수 설정
    print(f"Using loss function: {loss_function}...")
    if loss_function == "TripletLoss":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=triplet_margin
        )
    elif loss_function == "MultipleNegativesRankingLoss":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    
    # Evaluator 설정
    print("Setting up evaluator...")
    val_queries_df = val_df[val_df['pseudo_query'].notna()]

    queries = {str(i): q for i, q in zip(val_queries_df.index, val_queries_df['pseudo_query'])}
    corpus = {str(i): c for i, c in zip(val_df.index, val_df['combined_text'])}
    relevant_docs = {str(i): {str(i)} for i in val_queries_df.index}

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name='val_evaluator',
        show_progress_bar=False
    )

    # 학습 시작
    print(f"\nStarting training for {epochs} epochs...")

    if warmup_steps is None or warmup_steps <= 0:
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        steps_per_epoch=None,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        weight_decay=weight_decay,
        evaluation_steps=eval_steps,
        output_path=output_dir,
        save_best_model=save_best_model,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
        scheduler=scheduler,
        show_progress_bar=True
    )
    
    print(f"Training finished. Model saved to {output_dir}")
    
    return model