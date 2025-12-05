"""KaggleHub에서 와인 리뷰 데이터셋을 다운로드하는 스크립트"""
import os
import shutil
import kagglehub

def download_dataset():
    # data 디렉토리가 없으면 생성
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 최신 버전 다운로드
    print(f"\nDownloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("zynicide/wine-reviews")
    print(f"\nDownloaded to: {path}")

    # winemag-data-130k-v2.csv 파일을 data 디렉토리로 복사
    source_file = os.path.join(path, 'winemag-data-130k-v2.csv')
    dest_file = os.path.join(data_dir, 'winemag-data-130k-v2.csv')
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_file)
        print(f"Copied to: {dest_file}")
    else:
        raise FileNotFoundError(f"File not found: {source_file}")

    return dest_file