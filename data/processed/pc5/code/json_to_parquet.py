from pathlib import Path
import json
import pandas as pd

IN_DIR = Path("C:/Users/soldesk/Desktop/Jeongjs/workspace/MS Project/DUR-PREP/data/raw/dur_item")          # ✅ 9개 json 폴더
OUT_DIR = Path("C:/Users/soldesk/Desktop/Jeongjs/workspace/MS Project/DUR-PREP/data/processed/pc5/parts")  # ✅ 저장 위치
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_items(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "response" in raw:
        items = raw["response"]["body"]["items"]["item"]
        return items if isinstance(items, list) else [items]
    # fallback: dict 안에서 list 찾기
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

    # ✅ 숫자처럼 보여도 string으로 통일 (2E+07 같은 깨짐 방지)
    for c in df.columns:
        df[c] = df[c].astype("string")

    out_path = OUT_DIR / (p.stem + ".parquet")
    df.to_parquet(out_path, index=False)

    report.append((p.name, len(df), len(df.columns), out_path.as_posix()))

print("\n✅ PC5 완료 리포트")
for name, rows, cols, outp in report:
    print(f"- {name}: rows={rows:,}, cols={cols} -> {outp}")