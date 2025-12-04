import pandas as pd
import os

from src.preprocessing import preprocess_data


def read_csv(file_path):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    return pd.read_csv(file_path)


if __name__ == "__main__":

    # 데이터 불러오기
    csv_path = os.path.join('data', 'winemag-data-130k-v2.csv')
    df = read_csv(csv_path)

    # 데이터 전처리
    df_preprocessed = preprocess_data(df)
    print("Combined_text 예시:")
    print(df_preprocessed["combined_text"][0])

    
    