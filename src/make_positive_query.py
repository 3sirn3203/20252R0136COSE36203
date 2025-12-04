import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are an expert search assistant writing short, realistic queries that sound like a casual "
    "wine shopper (not an expert) trying to find these reviews."
)

STYLE_VARIANTS = [
    ("keyword", "Compact keyword-style query with grape/style, origin, and price hook."),
    ("natural", "Casual natural-language ask highlighting taste impression or vibe."),
    ("vague", "Looser vibe-driven description with one memorable hook (e.g., mineral, bright)."),
]


class LLMQueryGenerator:
    def __init__(
        self,
        model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        temperature: float = 0.7,
        max_tokens: int = 128,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        chosen_model = model_name or "unsloth/llama-3-8b-Instruct-bnb-4bit"
        if chosen_model == "llama-3.1-70b-versatile":
            print("model_name points to Groq remote default; using 'unsloth/llama-3-8b-Instruct-bnb-4bit' for local inference.")
            chosen_model = "unsloth/llama-3-8b-Instruct-bnb-4bit"

        self.model_name = chosen_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 환경이 필요합니다. GPU가 감지되지 않았습니다.")

        self.device = "cuda"
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        FastLanguageModel.for_inference(self.model)

    @staticmethod
    def _build_user_message(batch: List[Tuple[int, str]]) -> str:
        rows = []
        for idx, text in batch:
            rows.append(f"ID: {idx}\nREVIEW: {text.strip()}")

        style_lines = [f"- {name}: {desc}" for name, desc in STYLE_VARIANTS]

        prompt = (
            "Create one plausible search query for each wine review below.\n"
            "Guidelines:\n"
            "- Length: 5-25 tokens.\n"
            "- Tone: casual and curious, like a non-expert typing in a search bar; avoid long adjective lists.\n"
            "- Capture just 1-2 hints (grape/style + place or vibe); skip exhaustive tasting-note dumps.\n"
            "- Styles (pick ONE at random for each review, do not output which style):\n"
            f"{'\\n'.join(style_lines)}\n"
            "- Only one query per review (do NOT produce multiple variants).\n"
            "- No bullet lists, no explanations, no quotes.\n"
            'Return ONLY a JSON array of objects: {"id": <ID>, "query": "<user query>"} '
            "in the same order as the input IDs.\n\n"
            "Reviews:\n"
            f"{'\n\n'.join(rows)}"
        )
        return prompt

    @staticmethod
    def _extract_json(text: str) -> str:
        """LLM 응답에서 JSON 배열만 추출"""
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        array_match = re.search(r"(\[.*\])", text, re.DOTALL)
        if array_match:
            return array_match.group(1).strip()

        return text.strip()

    def _parse_response(self, content: str, expected_ids: Sequence[int]) -> Dict[int, str]:
        payload = self._extract_json(content)

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}

        results: Dict[int, str] = {}
        for item in data:
            try:
                idx = int(item.get("id"))
            except (TypeError, ValueError):
                continue

            if idx not in expected_ids:
                continue

            query = str(item.get("query", "")).strip()
            if query:
                results[idx] = query

        return results

    def generate_batch(self, batch: List[Tuple[int, str]]) -> Dict[int, str]:
        user_message = self._build_user_message(batch)

        inputs = self.tokenizer(
            [f"{SYSTEM_PROMPT}\n\n{user_message}"],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = generated[0, prompt_len:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        expected_ids = [idx for idx, _ in batch]
        parsed = self._parse_response(content, expected_ids)

        missing = set(expected_ids) - set(parsed.keys())
        if missing:
            print(f"[LLM] Missing queries for IDs: {sorted(list(missing))[:5]} (total {len(missing)})")

        return parsed

    def append_queries(
        self,
        df: pd.DataFrame,
        text_column: str = "combined_text",
        batch_size: int = 8,
        max_rows: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"DataFrame에 '{text_column}' 컬럼이 없습니다.")

        df_result = df.copy()
        if "pseudo_query" not in df_result.columns:
            df_result["pseudo_query"] = pd.NA

        working_df = df_result if max_rows is None else df_result.iloc[:max_rows]

        for start in range(0, len(working_df), batch_size):
            end = start + batch_size
            batch_df = working_df.iloc[start:end]
            payload = [(int(idx), str(text)) for idx, text in zip(batch_df.index, batch_df[text_column])]

            batch_queries = self.generate_batch(payload)
            for idx, query in batch_queries.items():
                df_result.at[idx, "pseudo_query"] = query
            if not start % (batch_size * 10):
                print(f"[LLM] Processed {min(end, len(working_df))}/{len(working_df)} reviews")

        if output_path:
            subset = df_result.loc[working_df.index, ["pseudo_query"]].reset_index()
            subset.rename(columns={"index": "row_id"}, inplace=True)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            subset.to_csv(output_path, index=False)

        return df_result
