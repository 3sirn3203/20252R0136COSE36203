"""KaggleHub에서 와인 리뷰 데이터셋을 다운로드하는 스크립트"""
import os
import kagglehub

def download_dataset():
    # data 디렉토리가 없으면 생성 및 다운로드 경로 설정
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    os.environ['KAGGLEHUB_CACHE'] = data_dir


    # 최신 버전 다운로드
    print(f"Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("zynicide/wine-reviews")
    print("Path to dataset files:", path)

    return path