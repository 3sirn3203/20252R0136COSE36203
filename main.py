import os
import json
import numpy as np
import pandas as pd
import argparse
import time
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.download_data import download_dataset
from src.preprocessing import preprocess_data
from src.make_positive_query import LLMQueryGenerator


load_dotenv()

DATA_PATH = "data/winemag-data-130k-v2.csv"
CONFIG_PATH = "src/config/config.json"
QUERY_PATH = "data/pseudo_queries.csv"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_json(file_path: str):
    """JSON 파일을 읽어 딕셔너리로 반환하는 함수"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_embedding_model(model_name: str ="all-mpnet-base-v2", device: str ="cpu") -> SentenceTransformer:
    """사전 학습된 문장 임베딩 모델을 로드하는 함수"""
    model = SentenceTransformer(model_name, device=device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="와인 리뷰 데이터셋 다운로드 및 전처리")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="config 파일 경로")
    args = parser.parse_args()

    config_path = args.config
    config = load_json(config_path)

    embedding_config = config.get("embedding", {})
    emb_model_name = embedding_config.get("model_name", "all-mpnet-base-v2")
    emb_batch_size = embedding_config.get("batch_size", 32)

    generate_config = config.get("generate_queries", {})
    enable_query_generation = generate_config.get("enable", True)
    output_path = generate_config.get("output_path", QUERY_PATH)
    gen_model_name = generate_config.get("model_name", "gemini-1.5-pro")
    gen_temperature = generate_config.get("temperature", 0.7)
    gen_max_tokens = generate_config.get("max_tokens", 512)
    gen_batch_size = generate_config.get("batch_size", 16)
    gen_max_rows = generate_config.get("max_rows", None)

    device = config.get("device", "cpu")

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
    print(f"\nTotal rows after preprocessing: {len(df_preprocessed)}")
    print("Example of combined_text:")
    print(f"{df_preprocessed['combined_text'].iloc[0]}")

    # LLM 기반 pseudo query 생성
    if enable_query_generation:
        q_start = time.time()
        print("\nGenerating pseudo queries with Gemini LLM...")
        llm_generator = LLMQueryGenerator(
            api_key=GEMINI_API_KEY,
            model_name=gen_model_name,
            temperature=gen_temperature,
            max_tokens=gen_max_tokens,
        )
        df_preprocessed = llm_generator.append_queries(
            df_preprocessed,
            text_column="combined_text",
            batch_size=gen_batch_size,
            max_rows=gen_max_rows,
            output_path=output_path,
        )
        q_end = time.time()
        elapsed = q_end - q_start
        print(f"Query generation elapsed time: {elapsed:.2f} seconds")

        first_idx = df_preprocessed["pseudo_query"].first_valid_index()
        if first_idx is not None:
            sample = df_preprocessed.loc[first_idx]
            print("\nExample pseudo query:")
            print(f"row_id={first_idx} | query={sample['pseudo_query']}")

    if os.path.exists(output_path):
        query = pd.read_csv(output_path)
        print(f"\nLoaded {len(query)} pseudo queries from {output_path}")


    # 임베딩 모델 로드
    # model = load_embedding_model(model_name=emb_model_name, device=device)

    # # 임베딩 계산
    # embeddings = model.encode(
    #     df_preprocessed["combined_text"].tolist(), 
    #     batch_size=emb_batch_size,
    #     show_progress_bar=True
    # )
    # embeddings = np.array(embeddings).astype(np.float32)

    # print(f"\nembeddings.shape: {embeddings.shape}")
