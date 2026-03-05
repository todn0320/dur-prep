import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import glob

# =========================
# 1. 환경 설정 및 출력 최적화
# =========================
pd.set_option('display.max_colwidth', None)
pd.set_option('display.unicode.east_asian_width', True)

BASE_DIR = Path(__file__).resolve().parents[3]
IN_DIR = BASE_DIR / "processed/pc3/parts"
OUT_DIR = BASE_DIR / "processed/pc4"
OUT_BY_TYPE = OUT_DIR / "clean_by_type"
OUT_BY_TYPE.mkdir(parents=True, exist_ok=True)

# 🎯 남길 컬럼 (필수 정보)
KEEP_COLS_INGR = [
    "ITEM_SEQ",      # 품목기준코드
    "MTRAL_NM",      # 성분명
    "QNT",           # 함량
    "INGD_UNIT_CD",  # 단위코드
    "MAIN_INGR_ENG"  # 영어 성분명
]

# =========================
# 2. 주성분 전용 정제 함수
# =========================

def clean_ingredient_df(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 정규화
    df.columns = [str(c).strip().upper() for c in df.columns]

    # [중요] 성분 관련 텍스트 데이터 정제
    text_cols = ["MTRAL_NM", "QNT", "INGD_UNIT_CD", "MAIN_INGR_ENG"]
    for col in text_cols:
        if col in df.columns:
            # 문자열 변환 및 공백 정규화 (Regex 사용)
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            # 불필요한 결측 문자열 처리
            df.loc[df[col].isin(['nan', 'None', '', 'NaN']), col] = pd.NA

    # ITEM_SEQ 코드 앞자리 0 보존
    if "ITEM_SEQ" in df.columns:
        df["ITEM_SEQ"] = df["ITEM_SEQ"].astype(str).str.strip()

    # 필요한 컬럼만 추출
    out = df[[c for c in KEEP_COLS_INGR if c in df.columns]].copy()
    
    # 성분명(MTRAL_NM)이 없는 행은 무의미하므로 제거
    out = out.dropna(subset=["MTRAL_NM"])

    return out

# =========================
# 3. 실행부
# =========================

def main():
    # 🔍 패턴에 맞는 모든 파일 찾기 (01, 02 등 모든 숫자 대응)
    pattern = "drugPrmsnInfo_의약품제품주성분_*.parquet"
    target_files = sorted(list(IN_DIR.glob(pattern)))

    if not target_files:
        print(f"❌ 입력 파일 없음: {IN_DIR} 내에 '{pattern}' 패턴의 파일이 없습니다.")
        return

    print(f"🔎 총 {len(target_files)}개의 파일을 찾았습니다.")

    for p in target_files:
        out_path = OUT_BY_TYPE / f"{p.stem}_clean.parquet"

        try:
            print(f"\n▶ [주성분 데이터] 정제 시작: {p.name}")
            
            # 대용량 파일 메모리 맵 로드
            table = pq.read_table(p, memory_map=True)
            df = table.to_pandas()
            
            print(f"   - 로드 완료 (원본 행 수: {len(df):,})")
            
            # 정제 실행
            cdf = clean_ingredient_df(df)

            # 저장 (zstd 고압축)
            cdf.to_parquet(
                out_path,
                engine='pyarrow',
                index=False,
                compression='zstd'
            )

            print(f"✅ 정제 완료: {out_path.name} | 결과 행 수: {len(cdf):,}")
            
            # 간단 검증 (데이터가 있는 경우만)
            if not cdf.empty and "MTRAL_NM" in cdf.columns:
                sample = cdf.iloc[cdf['MTRAL_NM'].str.len().argmax()]
                print(f"   [샘플 검증] {sample['MTRAL_NM'][:20]}... ({sample['QNT']} {sample['INGD_UNIT_CD']})")

        except Exception as e:
            print(f"⚠️ {p.name} 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()