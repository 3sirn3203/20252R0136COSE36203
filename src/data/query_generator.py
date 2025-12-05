import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import google.generativeai as genai
from google.api_core import exceptions
import pandas as pd


# --- Prompt Engineering: 페르소나 및 스타일 정의 ---
SYSTEM_PROMPT = (
    "You are an expert search assistant. "
    "Your goal is to reverse-engineer realistic user search queries from wine reviews. "
    "Think like a non-expert shopper typing into a search bar."
    "You MUST output valid JSON only."
)

STYLE_VARIANTS = [
    ("keyword", "Specific query searching for the brand name, vintage, or exact wine name (e.g., 'Stag's Leap 2013', 'Nicosia Vulka Bianco')."),
    ("natural", "Conversational, describing a taste or feeling (e.g., 'smooth red wine that tastes like chocolate')."),
    ("situation", "Occasion-based or vague request (e.g., 'good wine for steak dinner', 'gift for boss')."),
]

# --- Prompt Engineering: Few-shot Examples (생성 예시) ---
FEW_SHOT_EXAMPLES = """
Example 1:
Input ID: 101
Input Review: "This is a rigid, tannic wine that needs time. It offers heavy aromas of black coffee and dried sage."
Output JSON: {"id": 101, "query": "bold red wine with strong tannins"}

Example 2:
Input ID: 102
Input Review: "A bright, cheerful white wine with notes of lemon zest, green apple, and a touch of honey."
Output JSON: {"id": 102, "query": "refreshing white wine citrusy and sweet"}
"""

class LLMQueryGenerator:
    def __init__(self, gen_query_config: Dict, api_key: Optional[str] = None):

        self.gen_query_config = gen_query_config
        self.model_name = gen_query_config.get("model_name", "models/gemini-2.0-flash")
        self.temperature = gen_query_config.get("temperature", 0.7)
        self.max_tokens = gen_query_config.get("max_tokens", 1024)
        self.batch_size = gen_query_config.get("batch_size", 16)
        self.max_rows = gen_query_config.get("max_rows", None)
        self.output_path = gen_query_config.get("output_path", "data/pseudo_queries.csv")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
            },
        )

    def _build_user_message(self, batch: List[Tuple[int, str]]) -> str:
        rows = []
        for idx, text in batch:
            rows.append(f"ID: {idx}\nREVIEW: {text.strip()[:1000]}")

        style_lines = [f"- {name}: {desc}" for name, desc in STYLE_VARIANTS]

        prompt = (
            "Task: Generate ONE plausible search query for EACH wine review below.\n"
            "DO NOT just paraphrase the review. Imagine what a real user would actually type to find this wine.\n\n"
            "Guidelines:\n"
            "- Length: 10~20 words (keep it realistic).\n"
            "- Tone: Casual, curious, non-expert.\n"
            "- Styles (randomly mix these styles across the batch):\n"
            f"{chr(10).join(style_lines)}\n"
            "- Constraint: Output MUST be a valid JSON object with a key 'results' containing a list of objects.\n"
            "- IMPORTANT: You MUST include the 'id' field for each object to match the input.\n\n"
            f"[Generation Examples]\n{FEW_SHOT_EXAMPLES}\n\n"
            "[Target Reviews]\n"
            f"{chr(10).join(rows)}\n\n"
            "Output JSON:"
        )
        return prompt

    def _parse_response(self, content: str, expected_ids: Sequence[int]) -> Dict[int, str]:
        """
        [수정됨] ID가 없는 경우 순서대로 매핑하는 로직 추가
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("[LLM Error] JSON decoding failed.")
            return {}

        # 1. JSON 구조 정규화 (리스트 추출)
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 'results', 'queries', 'data' 등 다양한 키 대응
            for key in ["results", "queries", "data", "response"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            # 키를 못 찾았는데 단일 객체인 경우
            if not items and "query" in data:
                items = [data]

        results: Dict[int, str] = {}
        
        # 2. 매핑 로직 (ID 우선 확인 -> 없으면 인덱스 매핑)
        for i, item in enumerate(items):
            try:
                # 쿼리 텍스트 추출
                query = str(item.get("query", "")).strip()
                if not query:
                    continue

                # Case A: ID가 명시되어 있는 경우 (Best)
                if "id" in item:
                    idx = int(item["id"])
                    if idx in expected_ids:
                        results[idx] = query
                
                # Case B: ID가 누락된 경우 -> 순서대로 매핑 (Fallback)
                else:
                    if i < len(expected_ids):
                        idx = expected_ids[i]
                        results[idx] = query
            
            except (ValueError, TypeError):
                continue

        return results

    def generate_batch(self, batch: List[Tuple[int, str]]) -> Dict[int, str]:
        user_message = self._build_user_message(batch)
        expected_ids = [idx for idx, _ in batch]

        for attempt in range(3):
            try:
                response = self.model.generate_content(user_message)
                content = response.text or ""
                
                parsed = self._parse_response(content, expected_ids)

                # 파싱 결과가 있고, 요청한 개수의 70% 이상이면 성공으로 간주
                if parsed and len(parsed) >= int(len(expected_ids) * 0.7):
                    return parsed
                
                # 결과가 너무 적으면 실패 처리하고 재시도
                if attempt < 2:
                    print(f"[Retry] Attempt {attempt+1}: Too few results ({len(parsed)}/{len(expected_ids)}). Retrying...")
                    time.sleep(1)

            except exceptions.ResourceExhausted:
                wait_time = 10 * (attempt + 1)
                print(f"[Rate Limit] Quota exceeded. Sleeping for {wait_time} seconds...")
                time.sleep(wait_time)
            
            except Exception as e:
                print(f"[API Error] Attempt {attempt+1}: {e}")
                time.sleep(2)

        print(f"[Fail] Failed to generate for batch IDs: {expected_ids[:3]}...")
        return {}

    def append_queries(self, df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:
        
        df_result = df.copy()
        if "pseudo_query" not in df_result.columns:
            df_result["pseudo_query"] = pd.NA

        working_df = df_result if self.max_rows is None else df_result.iloc[:self.max_rows]
        total_rows = len(working_df)

        print(f"Starting generation for {total_rows} rows (Batch Size: {self.batch_size})...")
    
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        for start in range(0, total_rows, self.batch_size):
            end = start + self.batch_size
            batch_df = working_df.iloc[start:end]
            payload = [(int(idx), str(text)) for idx, text in zip(batch_df.index, batch_df[text_column])]

            batch_queries = self.generate_batch(payload)

            # 결과 매핑
            for idx, query in batch_queries.items():
                df_result.at[idx, "pseudo_query"] = query

            # 진행 상황 출력
            if (start // self.batch_size) % 5 == 0:
                print(f"[Progress] {min(end, total_rows)}/{total_rows} | Generated: {len(batch_queries)}/{len(payload)}")

            # 중간 저장
            if self.output_path:
                subset = df_result.loc[
                    df_result["pseudo_query"].notna(), 
                    ["pseudo_query"] 
                ].reset_index()
                subset.rename(columns={"index": "row_id"}, inplace=True)
                subset.to_csv(self.output_path, index=False)

            # Sleep Time
            time.sleep(3)

        print(f"\n[Complete] All tasks finished. Final data saved to {self.output_path}")
        return self.output_path