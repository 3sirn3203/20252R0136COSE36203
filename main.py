import pandas as pd
from dotenv import load_dotenv
import os

from src.download_data import download_dataset
from src.preprocessing import preprocess_data


load_dotenv()
DATA_PATH = os.getenv('DATA_PATH', 'data/datasets/zynicide/wine-reviews/versions/4/winemag-data-130k-v2.csv')


def read_csv(file_path):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    return pd.read_csv(file_path)


if __name__ == "__main__":

    # 데이터 파일이 존재하지 않으면 다운로드
    if not os.path.exists(DATA_PATH):
        download_dataset()
    
    # 데이터 불러오기
    df = read_csv(DATA_PATH)

    # 데이터 전처리
    df_preprocessed = preprocess_data(df)
    print("Combined_text 예시:")
    print(df_preprocessed["combined_text"][0])