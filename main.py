import os
import json
import numpy as np
import pandas as pd
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.download_data import download_dataset
from src.preprocessing import preprocess_data


load_dotenv()

DATA_PATH = "data/datasets/zynicide/wine-reviews/versions/4/winemag-data-130k-v2.csv"
CONFIG_PATH = "src/config/config.json"


def load_json(file_path: str):
    """JSON 파일을 읽어 딕셔너리로 반환하는 함수"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_csv(file_path: str):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    return pd.read_csv(file_path)

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

    model_name = config.get("embedding_model", "all-mpnet-base-v2")
    device = config.get("device", "cpu")

    # 데이터 파일이 존재하지 않으면 다운로드
    if not os.path.exists(DATA_PATH):
        download_dataset()
    
    # 데이터 불러오기
    df = read_csv(DATA_PATH)

    # 데이터 전처리
    df_preprocessed = preprocess_data(df)
    print("Example of combined_text:")
    print(df_preprocessed["combined_text"][0])

    # 임베딩 모델 로드
    model = load_embedding_model(model_name=model_name, device=device)

    # 임베딩 계산
    embeddings = model.encode(df_preprocessed["combined_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype(np.float32)

    print(f"embeddings.shape: {embeddings.shape}")