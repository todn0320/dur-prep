import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\soldesk\Desktop\새 폴더 (2)\dur-prep\data\processed")

PC3 = BASE_DIR / "pc3/parts"
PC2 = BASE_DIR / "pc2/clean_by_type"

file = list(PC3.glob("*의약품개요정보*.parquet"))[0]

df = pd.read_parquet(file)

print("원본 shape:", df.shape)
print("원본 컬럼:", df.columns)

# 컬럼 소문자
df.columns = df.columns.str.lower().str.strip()

# 문자열 정리
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# 중복 제거
df = df.drop_duplicates()

print("정제 후 shape:", df.shape)

save = PC2 / "PC2_clean_의약품개요정보.parquet"

df.to_parquet(save, index=False)

print("저장 완료:", save)