from sentence_transformers import util


def evaluate_biencoder_model(model, full_df, test_df, top_k=10, batch_size=256):
    """
    Bi-encoder 모델을 평가하는 함수
    """
    print(f"\n[Evaluation] Start Evaluation @ K={top_k}")
    
    # 1. Corpus Encoding
    print(f"1. Encoding Corpus ({len(full_df)} docs)...")
    corpus_embeddings = model.encode(
        full_df['combined_text'].tolist(), 
        batch_size=batch_size
    )
    
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
    
    # 3. Retrieval (평가 함수가 직접 검색 수행)
    print("3. Searching & Scoring...")
    hits = util.semantic_search(
        query_embeddings, 
        corpus_embeddings, 
        top_k=top_k,
        score_function=util.cos_sim
    )
    
    # 4. Score Calculation (Recall@K)
    correct_count = 0
    
    # full_df의 인덱스가 0~N으로 리셋되었다고 가정
    for i, (idx, row) in enumerate(query_subset.iterrows()):
        # idx: 정답 문서의 인덱스 (row_id)
        # hits[i]: i번째 쿼리의 검색 결과 [{'corpus_id': 123, 'score': 0.9}, ...]
        
        predicted_ids = [hit['corpus_id'] for hit in hits[i]]
        
        if idx in predicted_ids:
            correct_count += 1
            
    recall = correct_count / len(query_subset)
    print(f"\n📈 [Result] Recall@{top_k}: {recall:.4f}")
    
    return recall