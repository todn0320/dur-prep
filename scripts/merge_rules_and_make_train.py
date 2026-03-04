from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "output" / "clean_all"
OUT_DIR = ROOT / "output" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 각 파일에서 표준 컬럼으로 매핑 (없으면 빈 값)
COL_MAP = {
    # 표준컬럼 : 가능한 원본컬럼 후보들(우선순위)
    "dur_type": ["TYPE_NAME", "dur_type"],
    "item_seq": ["ITEM_SEQ", "item_seq"],
    "item_name": ["ITEM_NAME", "item_name"],
    "ingr_code": ["INGR_CODE", "ingr_code"],
    "ingr_name": ["INGR_NAME", "INGR_KOR_NAME", "ingr_name"],
    "entp_name": ["ENTP_NAME", "entp_name"],
    "prohbt_content": ["PROHBT_CONTENT", "prohbt_content"],
    "effect_name": ["EFFECT_NAME", "effect_name"],  # 효능군중복에 있을 수 있음
    "change_date": ["CHANGE_DATE", "change_date"],
    "notification_date": ["NOTIFICATION_DATE", "notification_date"],
}

STD_COLS = list(COL_MAP.keys())

def pick_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([""] * len(df))

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
    s = s.str.replace(r"[ \t]+", " ", regex=True)
    s = s.str.strip()
    return s

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = pd.DataFrame()
    for std, candidates in COL_MAP.items():
        out[std] = pick_col(df, candidates)

    # 타입 강제(코드류는 문자열)
    for c in ["item_seq", "ingr_code"]:
        out[c] = out[c].fillna("").astype(str).str.strip()

    # 텍스트 정리
    for c in ["dur_type", "item_name", "ingr_name", "entp_name", "prohbt_content", "effect_name"]:
        out[c] = normalize_text(out[c])

    # 어떤 파일에서 왔는지 추적
    out["source_file"] = path.name
    return out

def main():
    paths = sorted(IN_DIR.rglob("*.parquet"))
    if not paths:
        raise SystemExit(f"No parquet found in {IN_DIR}")

    frames = []
    for p in paths:
        df = load_one(p)
        frames.append(df)
        print("loaded", p.name, "rows=", len(df))

    all_df = pd.concat(frames, ignore_index=True)

    # 2) 너무 빈 행 제거(품목명/성분명/내용 모두 비면 제거)
    all_df = all_df[
        (all_df["item_name"] != "") |
        (all_df["ingr_name"] != "") |
        (all_df["prohbt_content"] != "")
    ].copy()

    # 3) 중복 제거 (보수적으로)
    all_df = all_df.drop_duplicates(subset=["dur_type", "item_seq", "ingr_code", "prohbt_content"])

    # 저장: 통합 규칙 테이블
    out_rules = OUT_DIR / "dur_rules_all.parquet"
    all_df.to_parquet(out_rules, index=False)
    print("saved", out_rules, "rows=", len(all_df))

    # 4) 학습용 텍스트 생성 (분류라면 label=dur_type)
    # 텍스트 템플릿: 보기 좋고 일관되게
    def make_text(r):
        parts = []
        if r["item_name"]: parts.append(f"품목명: {r['item_name']}")
        if r["ingr_name"]: parts.append(f"성분명: {r['ingr_name']}")
        if r["entp_name"]: parts.append(f"업체명: {r['entp_name']}")
        if r["effect_name"]: parts.append(f"효능군: {r['effect_name']}")
        if r["prohbt_content"]: parts.append(f"금기/주의내용: {r['prohbt_content']}")
        return " | ".join(parts)

    train = pd.DataFrame()
    train["text"] = all_df.apply(make_text, axis=1)
    train["label"] = all_df["dur_type"]

    # 빈 텍스트 제거
    train = train[train["text"].str.len() > 0].copy()

    out_train_csv = OUT_DIR / "train_dataset.csv"
    train.to_csv(out_train_csv, index=False, encoding="utf-8-sig")
    print("saved", out_train_csv, "rows=", len(train))

    # jsonl도 같이 (LLM/RAG/파인튜닝용)
    out_train_jsonl = OUT_DIR / "train_dataset.jsonl"
    train.to_json(out_train_jsonl, orient="records", lines=True, force_ascii=False)
    print("saved", out_train_jsonl)

if __name__ == "__main__":
    main()