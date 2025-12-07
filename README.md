# 와인 리뷰 검색/재순위화 파이프라인

## Project Overview
- Kaggle wine-reviews 데이터(`winemag-data-130k-v2.csv`)를 내려받아 정제한 뒤, 문장 템플릿으로 `combined_text` 컬럼을 만들고, pseudo query를 붙여 검색 학습 데이터로 만듭니다.
- Gemini 기반 LLM을 통해 pseudo query를 생성할 수 있으며(`config/generate_query.json`), 이미 생성된 CSV가 있다면 그대로 사용합니다.
- 단일 단계 Bi-encoder(`single-stage-retriever`)와 Bi-encoder + Cross-encoder 두 단계(`two-stage-retriever`)를 모두 지원하며, 선택한 설정에 따라 학습/평가 파이프라인을 자동으로 수행합니다.
- 평가 시 FAISS 인덱스 또는 bi+cross 재순위화를 이용해 Recall@K를 출력합니다.

## 실행 방법
1) Python 3 가상환경 생성 및 패키지 설치  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2) (선택) pseudo query 생성을 켜려면 `.env`에 `GEMINI_API_KEY=<your-key>`를 설정하고, `config/generate_query.json`의 `enable` 값을 `true`로 변경하세요.  
   생성된 쿼리는 기본적으로 `data/pseudo_queries.csv`에 저장됩니다.
3) 실행  
   ```bash
   # 기본 Bi-encoder 베이스라인 (config/biencoder_baseline.json)
   python main.py --config config/biencoder_baseline.json --gen-query config/generate_query.json

   # Triplet 학습 포함 Bi-encoder LTR
   python main.py --config config/biencoder_ltr.json --gen-query config/generate_query.json

   # Two-stage(Bi + Cross) 학습/평가
   python main.py --config config/crossencoder_ltr.json --gen-query config/generate_query.json
   ```
- 데이터셋이 없다면 자동으로 KaggleHub에서 다운로드되어 `data/winemag-data-130k-v2.csv`로 저장됩니다.
- 학습된 모델과 인덱스는 설정 값에 따라 `models/`와 `data/embeddings/` 하위에 저장됩니다.

## CLI 옵션
- `--config <path>`: 학습/평가 파이프라인 전체 설정 파일 경로. 모델 타입(`single-stage-retriever`, `two-stage-retriever`), 데이터 분할, 학습/평가 하이퍼파라미터 및 출력 경로를 정의합니다. 기본값 `config/biencoder_baseline.json`.
- `--gen-query <path>`: pseudo query 생성 설정 파일 경로. `enable`을 `true`로 하면 Gemini로 쿼리를 생성해 `output_path`에 저장하며, `model_name`, `temperature`, `batch_size`, `max_rows` 등을 제어합니다. 기본값 `config/generate_query.json`.
- `--local-test <bool>`: 로컬 테스트 플래그(현재 로직에서는 주로 구성 확인용). 기본값 `False`.


## MLM 파인튜닝 모듈 (./fine_tune.py) 사용법

```bash
python -m fine_tune --config config/fine_tune_mlm.json
```
