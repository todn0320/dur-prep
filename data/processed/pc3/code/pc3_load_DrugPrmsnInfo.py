from pathlib import Path
import json
import pandas as pd

IN_DIR = Path("C:/Users/soldesk/Desktop/dur-prep/낱알식별정보.json")     # ✅ 3개 json 폴더
OUT_DIR = Path("C:/Users/soldesk/Desktop/dur-prep/realdata")   # ✅ 저장 위치
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_items(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "response" in raw:
        items = raw["response"]["body"]["items"]["item"]
        return items if isinstance(items, list) else [items]
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                for v2 in v.values():
                    if isinstance(v2, list):
                        return v2
    raise ValueError("items 리스트를 찾지 못함")

report = []

for p in sorted(IN_DIR.glob("*.json")):
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except UnicodeDecodeError:
        with open(p, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)

    items = extract_items(raw)
    df = pd.json_normalize(items)

    for c in df.columns:
        df[c] = df[c].astype("string")

    out_path = OUT_DIR / (p.stem + ".parquet")
    df.to_parquet(out_path, index=False)

    # 관계성분 후보 컬럼 자동 탐색 (PM에게 공유)
    related_candidates = [c for c in df.columns if "relat" in c.lower() or "관계" in str(c)]

    report.append((p.name, len(df), len(df.columns), related_candidates, out_path.as_posix()))

print("\n✅ PC3 완료 리포트")
for name, rows, cols, rel, outp in report:
    print(f"- {name}: rows={rows:,}, cols={cols}, related_cols={rel} -> {outp}")