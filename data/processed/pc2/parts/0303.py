import pandas as pd
from pathlib import Path

# ==============================
# 경로 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent

PC3_PATH = BASE_DIR.parent / "pc3" / "parts"
PC2_CLEAN_PATH = BASE_DIR / "clean_by_type"

PC2_CLEAN_PATH.mkdir(exist_ok=True)

# ==============================
# 파일 불러오기
# ==============================

drug_file = PC3_PATH / "PC3_drugPrmsn.parquet"

df = pd.read_parquet(drug_file)

print("원본 데이터 수:", len(df))

# ==============================
# 필요한 컬럼만 선택
# ==============================

keep_columns = [
    "ITEM_SEQ",
    "ITEM_NAME",
    "ENTP_NAME",
    "ETC_OTC_CODE",
    "EE_DOC_DATA",
    "UD_DOC_DATA",
    "NB_DOC_DATA",
    "MAIN_ITEM_INGR",
    "INGR_NAME",
    "ATC_CODE",
    "ITEM_PERMIT_DATE"
]

df = df[keep_columns]

# ==============================
# 컬럼 이름 변경
# ==============================

df = df.rename(columns={
    "ITEM_SEQ": "item_seq",
    "ITEM_NAME": "item_name",
    "ENTP_NAME": "entp_name",
    "ETC_OTC_CODE": "etc_otc",
    "EE_DOC_DATA": "efficacy_text",
    "UD_DOC_DATA": "usage_text",
    "NB_DOC_DATA": "warning_text",
    "MAIN_ITEM_INGR": "main_ingr",
    "INGR_NAME": "ingr_name",
    "ATC_CODE": "atc_code",
    "ITEM_PERMIT_DATE": "permit_date"
})

# ==============================
# 텍스트 정리
# ==============================

for col in ["efficacy_text", "usage_text", "warning_text"]:
    df[col] = df[col].astype(str).str.replace("\n", " ").str.strip()

# ==============================
# 저장
# ==============================

save_path = PC2_CLEAN_PATH / "drug_clean.parquet"

df.to_parquet(save_path, index=False)

print("저장 완료:", save_path)