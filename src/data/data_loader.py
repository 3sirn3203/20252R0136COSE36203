import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit


def split_data_by_group(df, group_col='title', 
                        val_size=0.1, test_size=0.1, random_state=42):
    """Title을 기준으로 그룹이 섞이지 않게 train, val, test 세트로 분할하는 함수"""

    df = df.reset_index(drop=True)

    # train - val+test 분할
    test_val_ratio = val_size + test_size
    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
    train_idx, tmp_idx = next(splitter_1.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx]
    tmp_df = df.iloc[tmp_idx]

    # val - test 분할
    relative_test_size = test_size / test_val_ratio
    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=random_state)
    val_idx, test_idx = next(splitter_2.split(tmp_df, groups=tmp_df[group_col]))

    val_df = tmp_df.iloc[val_idx]
    test_df = tmp_df.iloc[test_idx]

    return train_df, val_df, test_df

def make_triplets(df_train, num_negatives=2, random_state=42):
    """
    Train 데이터셋에서 (Query, Positive, Negative_list) 형태의 학습 데이터를 생성하는 함수
    - Query가 있는 행: Anchor & Positive 역할
    - Query가 없는 행: Negative Pool 역할
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 데이터 분리
    labeled_df = df_train[df_train['pseudo_query'].notna()].copy()
    unlabeled_df = df_train[df_train['pseudo_query'].isna()].copy()
    
    # negative pair의 후보 pool 생성
    negative_pool = unlabeled_df['combined_text'].tolist()
    
    if len(negative_pool) < num_negatives:
        raise ValueError("Negative로 사용할 데이터(Query가 없는 데이터)가 부족합니다.")

    triplets = []
    
    # Triplet 생성
    for row in labeled_df.itertuples():
        anchor_query = row.pseudo_query
        positive_doc = row.combined_text
        
        # Negative Sampling (랜덤하게 num_negatives개 추출)
        negatives = random.sample(negative_pool, num_negatives)
        
        """
            query: str
            positive: str
            negatives: List[str]
        """
        triplets.append({
            'query': anchor_query,
            'positive': positive_doc,
            'negatives': negatives
        })
        
    return triplets