import pandas as pd, json
import os

with open("C:/Users/soldesk/Desktop/dur-prep/낱알식별정보.json", encoding='utf-8') as f:
    raw = json.load(f)
# ── ──구조사전진단
if isinstance(raw, dict) and 'response' not in raw:
    print(' Swagger ⚠️ — 표준구조가아닐수있음PC1 에게구조공유후진행')
# ── Swagger OpenAPI ──구조자동탐색
def extract_items(raw):
    # 케이스1: 최상위가리스트
    if isinstance(raw, list):
        return raw
    # 케이스2: Swagger 표준구조response > body > items > item
    if 'response' in raw:
        try:
            items = raw['response']['body']['items']['item']
            return items if isinstance(items, list) else [items]
        except (KeyError, TypeError):
            pass
    # 케이스3: 첫번째dict 키아래리스트탐색
    for v in raw.values():
        if isinstance(v, list): return v
        if isinstance(v, dict):
            for v2 in v.values():
                if isinstance(v2, list): return v2
    raise ValueError('item — 리스트를찾지못했습니다PC1(PM) 에게구조공유!')
items = extract_items(raw)
df = pd.json_normalize(items)

print('행수:', len(df))
print('컬럼수:', len(df.columns))

print('=== 결측률 TOP 10 ===')
print(df.isna().mean().sort_values(ascending=False).head(10).to_string())
# warning_type 관련 unique 값
type_col = 'TYPE_NAME' # ← 실제 컬럼명으로 수정
if type_col in df.columns:
 print('\n=== 유형 unique 값 ===')
 print(df[type_col].value_counts().to_string())
# 관계성분 컬럼 확인 (병용금기의 핵심!)
related_candidates = [c for c in df.columns if 'relat' in c.lower() or '관계' in str(c)]
print('\n 관계성분 후보 컬럼 :', related_candidates)
if related_candidates:
 print(df[related_candidates[0]].isna().sum(), '/ 전체:', len(df), '가 비어있음')
# 중복 확인
print('\n 중복 행수:', df.duplicated().sum())


df = df.fillna("").astype(str)

# 저장할 폴더 만들기 (경로 주의!)
os.makedirs('data/processed/pc3', exist_ok=True)

# 파켓으로 굽기 (에러 없이 싹 들어갑니다!)
df.to_parquet('C:/Users/soldesk/Desktop/dur-prep/realdata/낱알식별정보.parquet', index=False)

print(' ✅ 저장 완료: C:/Users/soldesk/Desktop/dur-prep/realdata/낱알식별정보.parquet')
print(f'행수: {len(df):,} / 컬럼수: {len(df.columns)}')