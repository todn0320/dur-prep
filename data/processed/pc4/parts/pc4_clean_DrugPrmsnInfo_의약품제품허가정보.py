import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# [설정] 출력 제한 해제 (텍스트가 잘리지 않고 콘솔에 찍히도록 함)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.unicode.east_asian_width', True)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[3]
IN_DIR = BASE_DIR / "processed/pc3/parts"
OUT_DIR = BASE_DIR / "processed/pc4"
OUT_BY_TYPE = OUT_DIR / "clean_by_type"
OUT_BY_TYPE.mkdir(parents=True, exist_ok=True)

# 🎯 AI 학습 핵심 피처 및 남길 컬럼
KEEP_COLS = [
    "ITEM_SEQ",         # 키값
    "ITEM_NAME",        # 제품명
    "ENTP_NAME",        # 업체명
    "ETC_OTC_CODE",     # 전문/일반 코드
    "EE_DOC_DATA",      # 효능효과 (AI 핵심)
    "UD_DOC_DATA",      # 용법용량 (AI 핵심)
    "NB_DOC_DATA",      # 주의사항 (AI 핵심)
    "MAIN_ITEM_INGR",   # 주성분
    "INGR_NAME",        # 성분명
    "ATC_CODE",         # 약물 분류 코드
    "CHANGE_DATE"       # 변경일자
]

def clean_docs_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 컬럼명 정규화
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    # 2. 텍스트 데이터 정제 (AI 가독성 최적화)
    text_target_cols = [
        "ITEM_NAME", "ENTP_NAME", "EE_DOC_DATA", 
        "UD_DOC_DATA", "NB_DOC_DATA", "MAIN_ITEM_INGR", "INGR_NAME"
    ]
    
    for col in text_target_cols:
        if col in df.columns:
            # 문자열화 -> 결측치 빈값 -> 연속공백/줄바꿈을 단일 공백으로
            # (NLP 토큰화 효율을 위해 불필요한 공백 제거)
            df[col] = df[col].astype(str).fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
            # 'nan' 문자열 제거
            df.loc[df[col].str.lower() == 'nan', col] = ""
    
    # 3. 필요한 컬럼만 추출 (버릴 컬럼은 자동으로 필터링됨)
    out = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    
    # 4. 필수값 필터링
    out = out.dropna(subset=["ITEM_SEQ", "ITEM_NAME"])
    
    return out

def main():
    target_file_name = "drugPrmsnInfo_의약품제품허가정보.parquet"
    p = IN_DIR / target_file_name
    out_path = OUT_BY_TYPE / f"{p.stem}_clean.parquet"

    if not p.exists():
        print(f"❌ 파일 없음: {p}")
        return

    try:
        print(f"▶ [허가정보 AI 피처링] 처리 시작: {p.name}")

        # 2GB 대용량 대응을 위한 Memory Map 읽기
        table = pq.read_table(p, memory_map=True)
        df = table.to_pandas()
        
        print(f"   - 로드 완료 (원본 행 수: {len(df):,})")
        
        cdf = clean_docs_df(df)

        # [저장 설정] 텍스트 데이터는 zstd 압축이 가장 효율적입니다.
        cdf.to_parquet(
            out_path,
            engine='pyarrow',
            index=False,
            compression='zstd',
            row_group_size=10000  # 개별 텍스트가 기므로 그룹 사이즈를 작게 조절
        )

        print(f"✅ 정제 및 저장 완료: {out_path.name}")
        print(f"📊 최종 행 수: {len(cdf):,}")

        # =========================
        # 검증: AI Feature 텍스트 유실 확인
        # =========================
        if "NB_DOC_DATA" in cdf.columns:
            # 가장 데이터가 긴 행을 찾아 샘플로 출력
            valid_rows = cdf[cdf['NB_DOC_DATA'] != ""]
            if not valid_rows.empty:
                sample_row = valid_rows.loc[valid_rows['NB_DOC_DATA'].str.len().idxmax()]
                print("\n--- [AI Feature 데이터 보존 검증] ---")
                print(f"제품명: {sample_row['ITEM_NAME']}")
                print(f"주의사항(NB) 길이: {len(sample_row['NB_DOC_DATA']):,}자")
                print(f"텍스트 샘플: {sample_row['NB_DOC_DATA'][:200]}...") 

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    main()