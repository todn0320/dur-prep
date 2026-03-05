import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# =========================
# 1. 환경 설정 및 출력 최적화
# =========================
pd.set_option('display.max_colwidth', None)  # 긴 성분명/제품명 짤림 방지
pd.set_option('display.unicode.east_asian_width', True)

BASE_DIR = Path(__file__).resolve().parents[3]
IN_DIR = BASE_DIR / "processed/pc3/parts"
OUT_DIR = BASE_DIR / "processed/pc4"
OUT_BY_TYPE = OUT_DIR / "clean_by_type"
OUT_BY_TYPE.mkdir(parents=True, exist_ok=True)

# 🎯 AI 학습 및 DUR 탐지용 핵심 컬럼
KEEP_COLS = [
    "ITEM_SEQ",          # 품목기준코드 (Join 키)
    "ITEM_NAME",         # 제품명
    "ENTP_NAME",         # 업체명
    "ITEM_PERMIT_DATE",  # 허가일
    "SPCLTY_PBLC",       # 전문/일반
    "PRDUCT_TYPE",       # 의약품 분류 (제형 등)
    "ITEM_INGR_NAME",    # 성분명 (DUR 탐지 핵심)
    "EDI_CODE",          # 보험코드
    "CANCEL_DATE"        # 취소일 (유효성 판단)
]

# =========================
# 2. 정제 함수
# =========================

def clean_master_df(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 정규화
    df.columns = [str(c).strip().upper() for c in df.columns]

    # [중요] 텍스트 데이터 정제 (학습 데이터 품질 향상)
    text_cols = ["ITEM_NAME", "ENTP_NAME", "SPCLTY_PBLC", "PRDUCT_TYPE", "ITEM_INGR_NAME"]
    for col in text_cols:
        if col in df.columns:
            # 문자열 변환 -> 양끝 공백 제거 -> 중간 불필요한 공백/줄바꿈 제거
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            df.loc[df[col].isin(['nan', 'None', '']), col] = pd.NA

    # 날짜 정제 (YYYY-MM-DD 형식 문자열로 보존)
    date_cols = ["ITEM_PERMIT_DATE", "CANCEL_DATE"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime('%Y-%m-%d')

    # 코드 데이터 (앞자리 0 보존)
    for col in ["ITEM_SEQ", "EDI_CODE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 필요한 컬럼만 추출 및 유효 데이터 필터링
    out = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    
    # 중복 제거 및 필수값(제품명) 체크
    out = out.drop_duplicates(subset=["ITEM_SEQ"], keep="last")
    out = out.dropna(subset=["ITEM_NAME"])

    return out

# =========================
# 3. 실행부
# =========================

def main():
    target_file_name = "drugPrmsnInfo_의약품제품허가목록.parquet"
    p = IN_DIR / target_file_name
    out_path = OUT_BY_TYPE / f"{p.stem}_clean.parquet"

    if not p.exists():
        print(f"❌ 입력 파일 없음: {p}")
        return

    try:
        print(f"▶ [마스터 데이터] 정제 시작: {p.name}")
        
        # 2GB 대응: Memory Map 방식으로 읽어 메모리 과부하 방지
        table = pq.read_table(p, memory_map=True)
        df = table.to_pandas()
        
        cdf = clean_master_df(df)

        # 저장 (zstd 압축으로 AI 학습용 데이터 세트 최적화)
        cdf.to_parquet(
            out_path,
            engine='pyarrow',
            index=False,
            compression='zstd'
        )

        print(f"✅ 정제 완료: {out_path.name} | 행 수: {len(cdf):,}")
        
        # 데이터 유실 검증 (가장 긴 성분명 기준)
        if "ITEM_INGR_NAME" in cdf.columns:
            sample = cdf.loc[cdf['ITEM_INGR_NAME'].str.len().idxmax()]
            print("\n--- [DUR 탐지용 성분 데이터 검증] ---")
            print(f"제품명: {sample['ITEM_NAME']}")
            print(f"성분명 길이: {len(str(sample['ITEM_INGR_NAME']))}자")
            print(f"내용: {sample['ITEM_INGR_NAME']}")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    main()