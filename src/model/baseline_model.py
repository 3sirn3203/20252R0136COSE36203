import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

class BaselineModel:
    def __init__(self, config: dict, device: str = "cuda"):
        """
        사전 학습된 문장 임베딩 모델을 로드하는 클래스
        학습(fit) 과정 없이 Inference만 수행합니다.
        """
        model_name = config.get("model_name", "all-mpnet-base-v2")
        batch_size = config.get("batch_size", 128)
        tok_k = config.get("tok_k", 10)

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.tok_k = tok_k
        self.device = device

    def encode(self, texts: list, batch_size: int = None) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 벡터로 변환 (Normalize=True 권장: 코사인 유사도 계산 위해)
        """
        # normalize_embeddings=True로 설정하면 내적(dot product)만으로 코사인 유사도 계산 가능
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size or self.batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            device=self.device
        )
        return embeddings


def create_model(config: dict, device: str) -> BaselineModel:
    """Config 설정을 받아 모델 생성"""
    embedding_config = config.get("embedding", {})
    
    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"
    
    return BaselineModel(config=embedding_config, device=device)