import torch
from sentence_transformers import SentenceTransformer


class BaselineModel:

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cuda"):
        """사전 학습된 문장 임베딩 모델을 로드하는 클래스"""
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list, batch_size: int =512):
        """텍스트 리스트를 임베딩 벡터로 변환하는 메서드"""
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    

def create_model(config: dict) -> BaselineModel:
    """설정 딕셔너리를 사용하여 BaselineModel 인스턴스를 생성하는 함수"""
    model_name = config.get("embedding", {}).get("model_name", "all-mpnet-base-v2")
    device = config.get("device", "cpu")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"
    
    model = BaselineModel(model_name=model_name, device=device)
    return model
