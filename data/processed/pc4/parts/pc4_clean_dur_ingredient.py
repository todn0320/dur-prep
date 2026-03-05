from pathlib import Path
import pandas as pd

# =========================
# 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]  # 프로젝트 루트 자동 인식
IN_DIR = BASE_DIR / "processed/pc3/parts"
OUT_DIR = BASE_DIR / "processed/pc4"
OUT_BY_TYPE = OUT_DIR / "clean_by_type"

OUT_BY_TYPE.mkdir(parents=True, exist_ok=True)

WARNING_ENUM = {
    "병용금기": 0,
    "특정연령대금기": 1,
    "특정연령금기": 1,
    "임부금기": 2,
    "노인주의": 3,
    "용량주의": 4,
    "투여기간주의": 5,
    "효능군중복": 6,
    "효능군중복주의": 6,
    "분할주의": 7,
    "서방정분할주의": 7,
}

DROP_COLS = {"CHART", "MIXTURE_CHART", "BIZRNO", "BAR_CODE"}


# =========================
# 유틸
# =========================

def safe_series(df, col):
    
    if col in df.columns:
        return df[col]
    return pd.Series([pd.NA] * len(df))


def norm_text(s):
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date


# =========================
# 핵심 정리 함수
# =========================

def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    df.columns = [str(c).strip() for c in df.columns]

    # 무거운 컬럼 제거
    drop_exist = [c for c in DROP_COLS if c in df.columns]
    if drop_exist:
        df = df.drop(columns=drop_exist)

    # TYPE_NAME → warning_type
    df["TYPE_NAME"] = norm_text(safe_series(df, "TYPE_NAME")).str.replace(" ", "", regex=False)
    print("###########################")
    print(df["TYPE_NAME"].value_counts())
    print("###########################")

    df["warning_type"] = df["TYPE_NAME"].map(WARNING_ENUM).fillna(99).astype("int64")

    # 성분명
    if "INGR_KOR_NAME" in df.columns:
        df["ingr_name_ko"] = df["INGR_KOR_NAME"].astype("string")
    elif "INGR_NAME" in df.columns:
        df["ingr_name_ko"] = df["INGR_NAME"].astype("string")
    else:
        df["ingr_name_ko"] = pd.NA

    # 공통 필드
    df["prohbt_content"] = norm_text(safe_series(df, "PROHBT_CONTENT"))
    df["notification_date"] = to_date(safe_series(df, "NOTIFICATION_DATE"))
    df["item_permit_date"] = to_date(safe_series(df, "ITEM_PERMIT_DATE"))

    # 코드류는 string
    for c in ["INGR_CODE", "ITEM_SEQ", "MIXTURE_INGR_CODE"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # 최소 출력 컬럼만 유지
    keep_cols = [
        "warning_type",
        "TYPE_NAME",
        "INGR_CODE",
        "ingr_name_ko",
        "ITEM_SEQ",
        "ITEM_NAME",
        "ENTP_NAME",
        "prohbt_content",
        "notification_date",
        "item_permit_date",
        "MIXTURE_INGR_CODE",
    ]

    out = pd.DataFrame({c: df[c] if c in df.columns else pd.NA for c in keep_cols})

    # 중복 제거
    key_cols = [c for c in ["warning_type", "INGR_CODE", "ITEM_SEQ", "MIXTURE_INGR_CODE", "prohbt_content"] if c in out.columns]
    out = out.drop_duplicates(subset=key_cols, keep="last")

    return out


# =========================
# 메인
# =========================

def main():

    if not IN_DIR.exists():
        print("❌ 입력 폴더 없음:", IN_DIR)
        return

    parquet_files = list(IN_DIR.glob("*.parquet"))

    if not parquet_files:
        print("❌ parquet 파일 없음:", IN_DIR)
        return

    cleaned_files = []

    for p in sorted(parquet_files):
        try:
            print("▶ 처리중:", p.name)
            df = pd.read_parquet(p)
            cdf = clean_df(df)

            out_path = OUT_BY_TYPE / f"{p.stem}_clean.parquet"
            cdf.to_parquet(out_path, index=False)

            cleaned_files.append(out_path)
            print("   저장:", out_path.name, "| rows:", len(cdf))

        except Exception as e:
            print("⚠️ 파일 실패:", p.name, "|", e)

    if not cleaned_files:
        print("❌ 정제된 파일 없음")
        return

    # 최종 합치기
    print("▶ 전체 합치기")
    big = pd.concat([pd.read_parquet(p) for p in cleaned_files], ignore_index=True)

    final_path = OUT_DIR / "dur_item_clean.parquet"
    big.to_parquet(final_path, index=False)

    print("✅ FINAL:", final_path)
    print("총 rows:", len(big))


if __name__ == "__main__":
    main()