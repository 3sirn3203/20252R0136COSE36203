import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class BaselineModel:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        사전 학습된 문장 임베딩 모델을 로드하는 클래스
        학습(fit) 과정 없이 Inference만 수행합니다.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=device)
        self.device = device

    def encode(self, texts: list, batch_size: int = 256,
               show_progress_bar: bool = True, normalize_embeddings: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 벡터로 변환 (Normalize=True 권장: 코사인 유사도 계산 위해)
        normalize_embeddings=True로 설정하면 내적(dot product)만으로 코사인 유사도 계산 가능
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            normalize_embeddings=normalize_embeddings,
            device=self.device
        )
        return embeddings


# 인터페이스 호환을 위해 train_triplets와 val_df 인자를 포함
def create_model(config: dict, train_triplets, val_df, device: str) -> BaselineModel:
    """
    Config 설정을 받아 모델 생성
    BaselineModel에서는 훈련 없이 pre-trained 모델을 바로 반환
    """
    model_name = config.get("model_name", "all-mpnet-base-v2")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"
    
    return BaselineModel(model_name=model_name, device=device)