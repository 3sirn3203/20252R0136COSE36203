import torch
from sentence_transformers import SentenceTransformer


# 인터페이스 호환을 위해 train_triplets와 val_df 인자를 포함
def create_model(config: dict, 
                 train_config: dict,
                 train_triplets, 
                 val_df, 
                 device: str, 
                 random_state: int) -> SentenceTransformer:
    """
    Config 설정을 받아 모델 생성
    BaselineModel에서는 훈련 없이 pre-trained 모델을 바로 반환
    """
    model_name = config.get("model_name", "all-mpnet-base-v2")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Using CPU instead.")
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)
    
    return model