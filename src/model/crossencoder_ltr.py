import os
import random
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    InputExample,
    util
)

from src.model.biencoder_ltr import create_model as train_bi_encoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TwoStageRetriever:
    """
    Bi-Encoder로 후보를 추리고(Retrieval), 
    Cross-Encoder로 순위를 재조정(Re-ranking)하는 Two-Stage 검색 모델
    """
    def __init__(self, bi_encoder: SentenceTransformer, cross_encoder: CrossEncoder):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.corpus_embeddings = None
        self.documents = []

    def index_corpus(self, documents: List[str], batch_size: int = 128):
        """전체 문서(Candidate Pool)를 임베딩하여 저장"""
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        self.corpus_embeddings = self.bi_encoder.encode(
            documents, 
            batch_size=batch_size, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def search(self, query: str, top_k_retrieval: int = 50, top_k_rerank: int = 10):
        """
        Two-Stage Search 실행
        1. Bi-Encoder: top_k_retrieval (50~100)
        2. Cross-Encoder: top_k_rerank (10)
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus is not indexed. Call index_corpus() first.")

        # 1. Bi-Encoder로 후보 문서 추출
        query_emb = self.bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        hits = util.semantic_search(query_emb, self.corpus_embeddings, top_k=top_k_retrieval)[0]

        candidate_indices = [hit['corpus_id'] for hit in hits]
        candidate_docs = [self.documents[idx] for idx in candidate_indices]

        # 2. Cross-Encoder로 후보 문서 재순위화
        cross_input = [[query, doc] for doc in candidate_docs]
        
        scores = self.cross_encoder.predict(cross_input)
        sorted_indices = np.argsort(scores)[::-1]

        # 최종 결과
        final_results = []
        for idx in sorted_indices[:top_k_rerank]:
            original_idx = candidate_indices[idx]
            final_results.append({
                "corpus_id": original_idx,
                "document": self.documents[original_idx],
                "bi_score": hits[idx]['score'],
                "cross_score": scores[idx]
            })

        return final_results
    

def train_cross_encoder(model_config: Dict, train_config: Dict, train_triplets, val_df, device: str = 'cuda'):
    """Cross-Encoder 학습 및 로드"""
    model_name = model_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_seq_length = model_config.get("max_seq_length", 256)
    num_labels = model_config.get("num_labels", 1)
    activation_function = model_config.get("activation_function", "Sigmoid")

    re_train = train_config.get("re_train", True)
    epochs = train_config.get("epochs", 3)
    learning_rate = train_config.get("learning_rate", 2e-5)
    scheduler = train_config.get("scheduler", "WarmupLinear")
    warmup_steps = train_config.get("warmup_steps", 1000)
    train_batch_size = train_config.get("train_batch_size", 16)
    pos_neg_ratio = train_config.get("pos_neg_ratio", 4)
    weight_decay = train_config.get("weight_decay", 0.01)
    max_grad_norm = train_config.get("max_grad_norm", 1.0)
    use_amp = train_config.get("use_amp", True)
    output_dir = train_config.get("output_dir", "models/crossencoder_stage2_trained")
    eval_steps = train_config.get("eval_steps", 500)
    save_best_model = train_config.get("save_best_model", True)

    # 기존 모델 로드 여부 확인
    if os.path.exists(output_dir) and not re_train:
        print(f"Loading existing Cross-Encoder from {output_dir}")
        return CrossEncoder(output_dir, device=device)

    # 모델 인스턴스 생성
    print(f"\nCreating crossencoder model '{model_name}' on device '{device}'...")
    model = CrossEncoder(
        model_name, 
        num_labels=num_labels, 
        max_length=max_seq_length, 
        device=device
    )

    # Activation Function 설정
    if activation_function == "Sigmoid":
        model.activation_fct = nn.Sigmoid()

    # 데이터 준비 (Triplet -> Pairs with Label 1/0)
    print(f"Preparing training data using Pos/Neg Ratio 1:{pos_neg_ratio}...")
    train_samples = []
    for t in train_triplets:
        # Positive sample (Label 1.0)
        train_samples.append(InputExample(texts=[t['query'], t['positive']], label=1.0))
        
        # Negative sample (Label 0.0)
        negs_to_use = t['negatives'][:pos_neg_ratio] 
        for neg in negs_to_use:
            train_samples.append(InputExample(texts=[t['query'], neg], label=0.0))

    train_dataloader = DataLoader(
        train_samples, 
        shuffle=True, 
        batch_size=train_batch_size
    )

    # Evaluator 생성
    print("Setting up evaluator...")
    val_samples = []

    # Validation 데이터 샘플링 (속도를 위해 일부만 사용)
    val_queries_df = val_df[val_df['pseudo_query'].notna()]
    sample_size = min(500, len(val_queries_df))
    val_subset = val_queries_df.sample(n=sample_size, random_state=42)
    all_docs = val_df['combined_text'].tolist()
    
    for _, row in val_subset.iterrows():
        q, p = row['pseudo_query'], row['combined_text']
        # Positive Pair
        val_samples.append(InputExample(texts=[q, p], label=1.0))
        # Negative Pair (Random Sampling)
        n = random.choice(all_docs)
        val_samples.append(InputExample(texts=[q, n], label=0.0))

    # Sentence Pairs와 Labels 분리하여 Evaluator에 전달
    sentence_pairs = [ex.texts for ex in val_samples]
    labels = [ex.label for ex in val_samples]

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=sentence_pairs,
        labels=labels,
        name='val_evaluator'
    )

    # 학습 시작
    print(f"Starting training for {epochs} epochs...")
    
    # Warmup steps 자동 계산 (설정값이 없으면 10%로 설정)
    if warmup_steps is None or warmup_steps <= 0:
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        optimizer_params={'lr': learning_rate},
        weight_decay=weight_decay,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
        output_path=output_dir,
        evaluation_steps=eval_steps,
        save_best_model=save_best_model,
        show_progress_bar=True
    )
    
    print(f"Training finished. Model saved to {output_dir}")
    return model


# model_loader 스크립트의 interface 함수
def create_model(model_config: dict, 
                 train_config: dict,
                 train_triplets, 
                 val_df, 
                 device: str, 
                 random_state: int) -> TwoStageRetriever:
    
    """
    Bi-Encoder와 Cross-Encoder를 모두 학습(또는 로드)시킨 후,
    통합 추론 객체(TwoStageRetriever)를 반환하는 함수
    """
    set_seed(random_state)

    bi_model_config = model_config.get("bi_encoder", {})
    bi_train_config = train_config.get("bi_encoder", {})
    cross_model_config = model_config.get("cross_encoder", {})
    cross_train_config = train_config.get("cross_encoder", {})
    
    # 1. Bi-Encoder 학습
    bi_model = train_bi_encoder(
        model_config=bi_model_config, 
        train_config=bi_train_config, 
        train_triplets=train_triplets, 
        val_df=val_df, 
        device=device,
        random_state=random_state
    )
    
    # 2. Cross-Encoder 학습
    cross_model = train_cross_encoder(
        model_config=cross_model_config, 
        train_config=cross_train_config, 
        train_triplets=train_triplets, 
        val_df=val_df, 
        device=device
    )
    
    # 3. 통합 모델 생성
    print("\n[Step 3] Building Two-Stage Retriever Pipeline...")
    model = TwoStageRetriever(bi_encoder=bi_model, cross_encoder=cross_model)
    
    return model