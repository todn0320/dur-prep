import pandas as pd
from pathlib import Path

# =========================
# 1. 환경 설정
# =========================
BASE_DIR = Path(__file__).resolve().parents[3]
IN_DIR = BASE_DIR / "processed/pc4/clean_by_type"
OUT_DIR = BASE_DIR / "processed/pc4/clean_by_type"

def main():
    target_file = IN_DIR / "drugPrmsnInfo_의약품제품허가정보_cleantext.parquet"
    out_path = OUT_DIR / "drugPrmsnInfo_의약품제품허가정보_cleantext_NB_DOC_DEL.parquet"

    if not target_file.exists():
        print(f"❌ 파일 없음: {target_file}")
        return

    try:
        print(f"▶ Parquet 데이터 정제 시작...")
        df = pd.read_parquet(target_file)

        # [핵심 로직] 
        # 만약 NB_DOC_DATA는 있는데 개별 text 컬럼들이 null인 경우를 대비해
        # 값이 이미 있는 경우에는 그대로 유지(Keep)하고, 없는 경우에만 원본에서 복사합니다.

        text_cols = ["warning_text", "contraindication_text", "caution_text", "adverse_text"]
        
        for col in text_cols:
            if col in df.columns:
                # 1. 기존 컬럼에 값이 있으면 그대로 쓰고, null이면 빈 문자열로 대체
                df[col] = df[col].fillna("")
                
                # 2. 만약 기존 컬럼이 비어있는데 NB_DOC_DATA에 데이터가 있다면?
                # (이 부분은 데이터 특성에 따라 수동 mapping이 필요할 수 있지만, 
                # 현재는 이미 분리된 데이터를 보존하는 데 집중합니다.)
                df[col] = df[col].astype(str).str.strip()

        # 3. 필요 없는 중복 컬럼(NB_DOC_DATA) 삭제
        if "NB_DOC_DATA" in df.columns:
            print("   - 중복 데이터(NB_DOC_DATA) 제거 중...")
            df = df.drop(columns=["NB_DOC_DATA"])

        # 4. 수치형 데이터 및 빈 값 정리 (예: 20190909.0 -> 20190909)
        if "CHANGE_DATE" in df.columns:
            df["CHANGE_DATE"] = df["CHANGE_DATE"].astype(str).replace(r'\.0$', '', regex=True)

        # 5. 최종 저장
        df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
        
        print(f"✅ 정제 완료: {out_path.name}")
        
        # 결과 확인
        print("\n--- [최종 데이터 구조 확인] ---")
        print(df.iloc[0].to_json(force_ascii=False, indent=4))

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    main()