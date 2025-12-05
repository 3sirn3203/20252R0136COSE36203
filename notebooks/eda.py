import pandas as pd
import os


def read_csv(file_path, index_col):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    return pd.read_csv(file_path, index_col=index_col)


if __name__ == "__main__":

    # 데이터 불러오기
    csv_path = os.path.join('data', 'winemag-data-130k-v2.csv')
    df = read_csv(csv_path, index_col='Unnamed: 0')

    # overview
    print("DataFrame Overview:")
    print(df.info())

    # 각 column별 결측값 비율
    print("\n결측값 비율:")
    missing_ratio = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, ratio in missing_ratio.items():
        print(f"{col}: {ratio:.2f}% ({df[col].isnull().sum()}/{len(df)})")

    # 정답 label (title) 기준으로 중복 데이터 확인
    # 중복이 있는 경우 test 
    duplicated_mask = df.duplicated(subset=['title'], keep=False)
    df_duplicated = df[duplicated_mask]
    df_unique = df[~duplicated_mask]
    
    # 결과 확인
    print(f"전체 데이터: {len(df)} rows")
    print(f"중복되는 title을 가진 rows: {len(df_duplicated)} rows")
    print(f"고유한 title을 가진 rows: {len(df_unique)} rows")