import pandas as pd

template = """
{description}
This is a {variety} wine from {winery}.
It is produced in {region_1}, {province}, {country}. 
The wine is designated as {designation} and received {points} points.
It is priced at ${price}. 
This wine was reviewed by {taster_name}.
"""

def _remove_duplicates(df):
    """중복 레코드 제거"""
    df_cleaned = df.drop_duplicates(subset='description', keep='first')
    return df_cleaned

def _remove_unnecessary_features(df):
    """불필요한 피처 제거 (taster_twitter_handle, region_2)"""
    columns_to_drop = ['taster_twitter_handle', 'region_2']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
    return df

def _clean_text(df):
    """텍스트 정제: 불필요한 공백 제거"""
    text_columns = ['description', 'designation', 'variety', 'country', 'province', 'region_1', 'winery', 'taster_name', 'title']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    return df

def _process_numeric_features(df, upper=1000):
    """수치형 데이터 처리: price 클리핑만 수행 (스케일링은 BERT용으로 제외)"""
    # price 클리핑 (상한값 1000)
    if 'price' in df.columns:
        df['price'] = df['price'].clip(upper=upper)
    
    return df

def _handle_missing_values(df):
    """결측값 처리"""
    # 수치형 피처: 중앙값으로 대체
    numeric_cols = ['price', 'points']
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    # 범주형 피처: "Unknown"으로 대체
    categorical_cols = ['country', 'designation', 'province', 'region_1', 'taster_name', 'variety', 'winery']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
    
    # description과 title은 빈 문자열로 대체
    text_cols = ['description', 'title']
    for col in text_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('')
    
    return df

def _make_combined_text(df):
    """DataFrame의 컬럼들을 template을 사용하여 combined_text로 생성"""
    
    # template을 사용하여 combined_text 생성
    df['combined_text'] = df.apply(
        lambda row: template.format(
            description=row['description'],
            variety=row['variety'],
            winery=row['winery'],
            region_1=row['region_1'],
            province=row['province'],
            country=row['country'],
            designation=row['designation'],
            points=row['points'],
            price=row['price'],
            taster_name=row['taster_name']
        ).strip(),
        axis=1
    )
    
    return df

def preprocess_data(df):
    """데이터 전처리 함수"""
    
    # 1. 중복 제거
    df = _remove_duplicates(df)
    
    # 2. 불필요 피처 제거
    df = _remove_unnecessary_features(df)
    
    # 3. 결측값 처리 (텍스트 정제 전에 수행)
    df = _handle_missing_values(df)
    
    # 4. 텍스트 정제
    df = _clean_text(df)
    
    # 5. 수치형 데이터 처리 (클리핑만 수행, BERT용으로 스케일링 제외)
    df = _process_numeric_features(df, upper=1000)
    
    # 6. combined_text 생성
    df = _make_combined_text(df)
    
    return df