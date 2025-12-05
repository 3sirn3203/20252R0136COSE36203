import os
import json
import pandas as pd
import argparse
import time
import warnings
import torch
from dotenv import load_dotenv

from src.data.downloader import download_dataset
from src.data.preprocessing import preprocess_data
from src.data.data_loader import split_data_by_group, make_triplets
from src.model_loader import load_model
from src.evaluate import evaluate_biencoder_model
from src.data.query_generator import LLMQueryGenerator


load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')

DATA_PATH = "data/winemag-data-130k-v2.csv"
CONFIG_PATH = "config/baseline_config.json"
QUERY_PATH = "data/pseudo_queries.csv"


def load_json(file_path: str):
    """JSON 파일을 읽어 딕셔너리로 반환하는 함수"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="와인 리뷰 데이터셋 다운로드 및 전처리")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="config 파일 경로")
    parser.add_argument("--local-test", type=bool, default=False, help="로컬 테스트 모드 여부")
    args = parser.parse_args()

    config_path = args.config
    config = load_json(config_path)

    gen_query_config = config.get("generate_queries", {})
    enable_query_generation = gen_query_config.get("enable", True)

    data_config = config.get("data", {})
    data_path = data_config.get("path", DATA_PATH)
    embedding_path = data_config.get("embedding_path", None)
    query_path = data_config.get("query_path", QUERY_PATH)
    val_size = data_config.get("val_size", 0.1)
    test_size = data_config.get("test_size", 0.1)
    num_negatives = data_config.get("num_negatives", 2)

    model_config = config.get("model", {})
    evaluate_config = config.get("evaluate", {})

    device = config.get("device", "cpu")
    random_state = config.get("random_state", 42)

    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"

    # 데이터 파일이 존재하지 않으면 다운로드
    if not os.path.exists(DATA_PATH):
        download_dataset()
    
    # 데이터 불러오기
    df = pd.read_csv(DATA_PATH)

    # 데이터 전처리
    df_preprocessed = preprocess_data(df)
    df_preprocessed = df_preprocessed.reset_index(drop=True)
    print("\nExample of combined_text:")
    print(f"{df_preprocessed['combined_text'].iloc[0]}")

    # LLM 기반 pseudo query 생성
    if enable_query_generation:
        # API 키 확인
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
        
        q_start = time.time()
        print("\nGenerating pseudo queries with Gemini LLM...")
        llm_generator = LLMQueryGenerator(
            gen_query_config=gen_query_config,
            api_key=GEMINI_API_KEY,
        )
        query_path = llm_generator.append_queries(
            df_preprocessed,
            text_column="combined_text"
        )
        q_end = time.time()
        elapsed = q_end - q_start
        print(f"Query generation elapsed time: {elapsed:.2f} seconds")

        first_idx = df_preprocessed["pseudo_query"].first_valid_index()
        if first_idx is not None:
            sample = df_preprocessed.loc[first_idx]
            print("\nExample pseudo query:")
            print(f"row_id={first_idx} | query={sample['pseudo_query']}")

    # 이미 생성된 pseudo query 불러오기
    if os.path.exists(query_path):
        query = pd.read_csv(query_path)
        print(f"\nLoaded {len(query)} pseudo queries from {query_path}")

    # pseudo query를 DataFrame에 추가
    df_preprocessed = df_preprocessed.merge(query, left_index=True, right_index=True, how='left')
    
    print(f"\nFinal DataFrame shape: {df_preprocessed.shape}")
    print(f"Columns: {df_preprocessed.columns.tolist()}")

    # train, val, test 세트로 분할
    train_df, val_df, test_df = split_data_by_group(
        df_preprocessed,
        group_col='title',
        val_size=val_size,
        test_size=test_size,
        random_state=random_state
    )

    print("\nDataset sizes:")
    print(f"  Train set size: {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")

    # pseudo_query가 있는 row 개수 확인
    train_with_query = train_df['pseudo_query'].notna().sum()
    val_with_query = val_df['pseudo_query'].notna().sum()
    test_with_query = test_df['pseudo_query'].notna().sum()
    
    print(f"\nRows with pseudo_query:")
    print(f"  Train: {train_with_query} / {len(train_df)} ({train_with_query/len(train_df)*100:.2f}%)")
    print(f"  Validation: {val_with_query} / {len(val_df)} ({val_with_query/len(val_df)*100:.2f}%)")
    print(f"  Test: {test_with_query} / {len(test_df)} ({test_with_query/len(test_df)*100:.2f}%)")

    # Data leakage 검사
    train_titles = set(train_df['title'])
    test_titles = set(test_df['title'])
    assert len(train_titles.intersection(test_titles)) == 0, "Data leakage detected between train and test sets!"

    train_triplets = make_triplets(
        train_df, 
        num_negatives=num_negatives, 
        random_state=random_state
    )

    print(f"\nGenerated {len(train_triplets)} training triplets.")
    print("Example Triplet:")
    print(f"  Query: {train_triplets[0]['query']}")
    print(f"  Positive: {train_triplets[0]['positive'][:50]}...")
    print(f"  Negative 1: {train_triplets[0]['negatives'][0][:50]}...")
    print(f"  Negative 2: {train_triplets[0]['negatives'][1][:50]}...")

    model = load_model(
        model_config=model_config, 
        train_triplets=train_triplets,
        val_df=val_df,
        device=device,
    )

    evaluate_biencoder_model(
        model=model,
        embedding_path=embedding_path,
        full_df=df_preprocessed,
        test_df=test_df,
        evaluate_config=evaluate_config,
    )