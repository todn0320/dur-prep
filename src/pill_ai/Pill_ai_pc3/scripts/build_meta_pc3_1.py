from pathlib import Path
import pandas as pd

# 이미지 폴더
IMG_ROOT = Path("src/pill_ai/Pill_ai_pc3/data/sample_img")

# 저장할 meta.csv 경로
OUT_CSV = Path("src/pill_ai/Pill_ai_pc3/data/meta/meta.csv")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []

for class_dir in sorted(IMG_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    class_id = class_dir.name

    image_files = sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    )

    for count, img_path in enumerate(image_files, start=1):
        rows.append({
            "image_id": f"{class_id}_{count:04d}",
            "image_path": str(img_path),
            "class_id": class_id,
            "side": "unknown",
            "print_gt": ""
        })

df = pd.DataFrame(rows)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("meta.csv 생성 완료")
print("총 이미지:", len(df))
print("클래스 수:", df["class_id"].nunique())
print(df.head(10))