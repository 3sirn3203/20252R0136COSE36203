import os
import importlib.util


def load_model(model_config: dict, train_triplets, val_df, device: str):
    """설정 딕셔너리를 사용하여 모델을 로드하는 함수"""
    model_path = model_config.get("path", "models/baseline.py")
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 동적으로 모듈 로드
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {model_path}")
    
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    # create_model 함수 존재 확인
    if not hasattr(model_module, 'create_model'):
        raise AttributeError(f"'create_model' function not found in {model_path}")
    
    # create_model 함수 호출하여 모델 생성
    model = model_module.create_model(model_config, train_triplets, val_df, device)
    
    return model    