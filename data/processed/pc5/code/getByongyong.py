# 병용금기 데이터가 41만개 이상이라 3만개씩 나눠서 Parquet로 저장

from pathlib import Path
import pandas as pd, json

with open("C:/Users/soldesk/Desktop/Jeongjs/workspace/MS Project/DUR-PREP/data/raw/dur_item/dur_item_병용금기.json", encoding='utf-8') as f:
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

OUT_DIR = Path("C:/Users/soldesk/Desktop/Jeongjs/workspace/MS Project/DUR-PREP/data/processed/pc5/parts/dur_item_병용금기")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP = [
    "DUR_SEQ","TYPE_NAME","MIX","MIX_TYPE",
    "INGR_CODE","INGR_KOR_NAME","INGR_ENG_NAME","MIX_INGR",
    "ITEM_SEQ","ITEM_NAME","ENTP_NAME","FORM_NAME","ETC_OTC_NAME","CLASS_NAME","MAIN_INGR",
    "MIXTURE_DUR_SEQ","MIXTURE_MIX",
    "MIXTURE_INGR_CODE","MIXTURE_INGR_KOR_NAME","MIXTURE_INGR_ENG_NAME",
    "MIXTURE_ITEM_SEQ","MIXTURE_ITEM_NAME","MIXTURE_ENTP_NAME",
    "MIXTURE_FORM_NAME","MIXTURE_ETC_OTC_NAME","MIXTURE_CLASS_NAME","MIXTURE_MAIN_INGR",
    "NOTIFICATION_DATE","PROHBT_CONTENT","REMARK","CHANGE_DATE","MIXTURE_CHANGE_DATE",
]

# 존재하는 컬럼만
KEEP = [c for c in KEEP if c in df.columns]
thin = df[KEEP].copy()

# string 강제(코드 깨짐 방지)
for c in thin.columns:
    thin[c] = thin[c].astype("string")

CHUNK = 30_000  #3만개씩
n = len(thin)
for i, start in enumerate(range(0, n, CHUNK), start=1):
    part = thin.iloc[start:start+CHUNK]
    out = OUT_DIR / f"part_{i:05d}.parquet"
    part.to_parquet(out, index=False, compression="zstd")  # 용량 줄이기
    print("saved", out, "rows", len(part))



    