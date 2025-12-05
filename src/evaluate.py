import os
import faiss
import numpy as np
from typing import Dict
from sentence_transformers import util

def evaluate_biencoder_model(model, full_df, test_df, evaluate_config: Dict):
    
    """Bi-encoder 모델 평가 함수"""
    top_k = evaluate_config.get("top_k", 10)
    batch_size = evaluate_config.get("batch_size", 256)
    embedding_path = evaluate_config.get("embedding_path", None)
    
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"\n[Evaluation] Start Evaluation @ K={top_k}")
    
    # 1. Corpus Indexing (Load Cache or Compute & Save)
    index = None
    
    if embedding_path:
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

    # 캐시된 인덱스가 존재하는 경우 불러옴
    if embedding_path and os.path.exists(embedding_path):
        print(f"1. Loading existing FAISS index from {embedding_path}...")
        try:
            # CPU 버전 FAISS 인덱스 로드
            index = faiss.read_index(embedding_path)
            print(f"   Successfully loaded index containing {index.ntotal} vectors.")
        except Exception as e:
            print(f"   [Error] Failed to load index: {e}. Recomputing...")
            index = None

    # 캐시된 인덱스가 없으면 계산 및 저장
    if index is None:
        print(f"1. Computing Corpus Embeddings ({len(full_df)} docs)...")
        
        corpus_embeddings = model.encode(
            full_df['combined_text'].tolist(), 
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        corpus_embeddings = corpus_embeddings.astype(np.float32)
        
        # FAISS 인덱스 생성 및 추가
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(corpus_embeddings)
        
        # 저장
        if embedding_path:
            print(f"   Saving FAISS index to {embedding_path}...")
            faiss.write_index(index, embedding_path)
    
    # 2. Query Encoding
    query_subset = test_df[test_df['pseudo_query'].notna()]
    if len(query_subset) == 0:
        return 0.0

    print(f"2. Encoding Queries ({len(query_subset)} queries)...")
    query_embeddings = model.encode(
        query_subset['pseudo_query'].tolist(), 
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    query_embeddings = query_embeddings.astype(np.float32)
    
    # 3. Retrieval (FAISS Search)
    print("3. Searching & Scoring...")
    
    # D: Distances (유사도 점수), I: Indices (문서 ID)
    D, I = index.search(query_embeddings, top_k)
    
    # 4. Score Calculation (Recall@K)
    correct_count = 0
    query_indices = query_subset.index.tolist()
    
    for i, true_doc_id in enumerate(query_indices):

        predicted_ids = I[i]
        
        if true_doc_id in predicted_ids:
            correct_count += 1
            
    recall = correct_count / len(query_subset)
    print(f"\n📈 [Result] Recall@{top_k}: {recall:.4f}")
    
    return recall