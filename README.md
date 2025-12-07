# 와인 추천/검색 시스템

## 프로젝트 목적
- Kaggle wine-reviews 데이터셋을 전처리하고, 검색/추천 파이프라인을 구축하여 사용자의 와인 질의에 맞는 문서(리뷰)를 추천합니다.
- Bi-encoder 기반의 단일 단계 검색과 Bi-encoder + Cross-encoder 재순위화(two-stage)를 모두 지원하여 품질과 속도를 선택적으로 조정할 수 있습니다.
- 필요 시 LLM(Gemini)으로 pseudo query를 생성하거나, MLM 파인튜닝을 통해 도메인 특화 임베딩을 개선합니다.

## 주요 특징
- 데이터 자동 다운로드 및 정제: KaggleHub로 원본을 내려받고, 템플릿 기반 `combined_text` 생성.
- Pseudo query 생성(옵션): Gemini로 사용자 검색 질의를 합성하여 학습 신호 강화.
- 다양한 설정: 베이스라인 Bi-encoder, Triplet 기반 학습, Two-stage(Bi + Cross) 조합을 config로 전환.
- 평가: FAISS 인덱스 기반 Recall@K 계산 또는 bi+cross 재순위화 평가.
- 추가 학습: `fine_tune.py`로 masked language modeling 파인튜닝 지원.

## 환경 준비 및 실행
1) 가상환경 + 의존성 설치 (Python 3.10+ 권장)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2) 환경 변수
   - Pseudo query를 생성하려면 `.env`에 `GEMINI_API_KEY=<your-key>`를 설정하고 `config/generate_query.json`의 `enable`을 `true`로 변경.
   - KaggleHub 사용 시 Kaggle 계정 인증이 필요할 수 있습니다.
3) 데이터 준비
   - `data/winemag-data-130k-v2.csv`가 없으면 `main.py`/`fine_tune.py` 실행 시 자동 다운로드됩니다.

### main.py (검색/추천 파이프라인)
```bash
# 베이스라인 Bi-encoder
python main.py --config config/biencoder_baseline.json --gen-query config/generate_query.json

# Triplet 학습 Bi-encoder LTR
python main.py --config config/biencoder_ltr.json --gen-query config/generate_query.json

# Two-stage (Bi + Cross) 재순위화
python main.py --config config/crossencoder_ltr.json --gen-query config/generate_query.json
```
- 출력: 학습된 모델(`models/...`), FAISS 인덱스(`data/embeddings/...`), 생성된 pseudo query(`data/pseudo_queries.csv`), 평가 로그.

### fine_tune.py (MLM 파인튜닝)
```bash
python fine_tune.py --config config/fine_tune_mlm.json
```
- 출력: 파인튜닝된 MLM 모델이 `models/fine_tuned_mlm`에 저장됩니다. SentenceTransformer로 로드 가능.

## CLI 옵션 정리
- `main.py`
  - `--config <path>`: 파이프라인 설정(JSON). 모델 타입, 데이터 분할, 학습/평가 하이퍼파라미터, 출력 경로를 정의. 기본 `config/biencoder_baseline.json`.
  - `--gen-query <path>`: pseudo query 생성 설정(JSON). `enable`, `model_name`, `temperature`, `batch_size`, `max_rows`, `output_path` 등을 제어. 기본 `config/generate_query.json`.
  - `--local-test <bool>`: 로컬 테스트 플래그(주로 구성 확인용). 기본 `False`.
- `fine_tune.py`
  - `--config <path>`: MLM 파인튜닝 설정(JSON). 모델/배치/에폭/학습률, `mlm_probability`, 로깅/저장 전략, `output_dir` 등을 포함. 기본 `config/fine_tune_mlm.json`.
  - 참고: `fine_tune.py`는 `data.path` 키를 우선 사용하고, 없으면 `data/winemag-data-130k-v2.csv`를 기본값으로 씁니다. 커스텀 경로를 쓰려면 `data.path`를 명시하세요.

## 프로젝트 구조
```
.
├─ main.py                     # 검색/추천 전체 파이프라인 엔트리포인트
├─ fine_tune.py                # MLM 파인튜닝 엔트리포인트
├─ config/
│  ├─ biencoder_baseline.json
│  ├─ biencoder_ltr.json
│  ├─ crossencoder_ltr.json
│  ├─ fine_tune_mlm.json
│  └─ generate_query.json
├─ src/
│  ├─ data/
│  │  ├─ downloader.py         # KaggleHub 데이터 다운로드
│  │  ├─ preprocessing.py      # 텍스트/수치 정제 및 combined_text 생성
│  │  ├─ data_loader.py        # 그룹 단위 분할, triplet 생성
│  │  └─ query_generator.py    # Gemini 기반 pseudo query 생성
│  ├─ model/
│  │  ├─ biencoder_baseline.py # 사전학습 Bi-encoder 로드
│  │  ├─ biencoder_ltr.py      # Triplet 기반 Bi-encoder 학습
│  │  └─ crossencoder_ltr.py   # Bi + Cross two-stage 학습/검색
│  ├─ model_loader.py          # 설정에 따라 모델 모듈 동적 로딩
│  └─ evaluate.py              # FAISS 검색 및 two-stage 평가
├─ data/                       # 원본/전처리 데이터, pseudo query, 인덱스 저장
├─ models/                     # 학습/파인튜닝된 모델 저장
├─ notebooks/                  # EDA 및 실험 노트북
└─ requirements.txt
```

## 스크립트/모듈 역할 (파이프라인 흐름 기준)
- `main.py`: 설정 로드 → 데이터 다운로드/전처리 → (옵션) pseudo query 생성/로딩 → train/val/test 분할 → triplet 생성 → 모델 로드·학습 → FAISS 혹은 two-stage 평가.
- `fine_tune.py`: 전처리된 `combined_text`로 MLM 파인튜닝을 수행해 도메인 특화 임베딩을 생성.
- `src/data/downloader.py`: KaggleHub에서 `winemag-data-130k-v2.csv`를 다운로드 후 `data/`에 복사.
- `src/data/preprocessing.py`: 중복 제거, 결측/텍스트 정제, 가격 클리핑, 템플릿 기반 `combined_text` 생성.
- `src/data/data_loader.py`: 제목 그룹 기준으로 분할(GroupShuffleSplit)하고, pseudo query가 없는 문서를 네거티브 풀로 삼아 triplet을 만듭니다.
- `src/data/query_generator.py`: Gemini LLM으로 배치 단위 pseudo query를 생성 후 CSV에 저장.
- `src/model_loader.py`: config의 `model.path`에 맞춰 모델 모듈을 동적 로드하고 `create_model`을 호출.
- `src/model/biencoder_baseline.py`: 추가 학습 없이 사전학습 SentenceTransformer를 로드해 바로 평가/검색.
- `src/model/biencoder_ltr.py`: TripletLoss 또는 MultipleNegativesRankingLoss로 Bi-encoder를 학습하며, val IR evaluator를 사용해 중간 평가/모델 저장.
- `src/model/crossencoder_ltr.py`: Bi-encoder 학습 후 Cross-encoder를 분리 학습해 TwoStageRetriever로 묶어 검색+재순위화를 제공합니다.
- `src/evaluate.py`: Bi-encoder는 FAISS 인덱스 기반 Recall@K, two-stage는 bi 검색 후 cross 재순위화를 거쳐 Recall@K를 계산.

## 추가 안내
- GPU 사용 시 `device`를 `cuda`로 설정하면 속도가 크게 개선됩니다(가용성 자동 체크).
- `config/generate_query.json`의 `enable=false`이면 기존 pseudo query CSV를 그대로 사용합니다.
- huggingface/Google API 호출 시 네트워크 사용이 필요하므로 방화벽/프로젝트 설정을 확인하세요.
- 실험 로그와 출력물은 config의 `output_dir`와 `embedding_path`에 저장되니, 중복 실행 시 덮어쓰기 여부를 확인하세요.
