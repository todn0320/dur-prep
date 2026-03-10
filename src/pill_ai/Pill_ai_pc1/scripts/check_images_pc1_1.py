from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

META_CSV = Path("src/pill_ai/Pill_ai_pc1/data/meta/meta.csv")
REPORT_DIR = Path("src/pill_ai/Pill_ai_pc1/data/meta")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META_CSV)

ok_rows = []
bad_rows = []

for _, row in df.iterrows():
    img_path = Path(row["image_path"])
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            mode = img.mode

        ok_rows.append({
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "class_id": row["class_id"],
            "width": w,
            "height": h,
            "mode": mode
        })
    except Exception as e:
        bad_rows.append({
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "class_id": row["class_id"],
            "error": str(e)
        })

ok_df = pd.DataFrame(ok_rows)
bad_df = pd.DataFrame(bad_rows)

ok_df.to_csv(REPORT_DIR / "image_open_ok.csv", index=False, encoding="utf-8-sig")
bad_df.to_csv(REPORT_DIR / "image_open_bad.csv", index=False, encoding="utf-8-sig")

count_df = df.groupby("class_id").size().reset_index(name="image_count")
count_df.to_csv(REPORT_DIR / "class_count.csv", index=False, encoding="utf-8-sig")

print("=== 로딩 테스트 완료 ===")
print("전체 이미지 수:", len(df))
print("정상 이미지 수:", len(ok_df))
print("깨진 이미지 수:", len(bad_df))
print("\n클래스별 개수:")
print(count_df)

if len(bad_df) > 0:
    print("\n깨진 파일 목록:")
    print(bad_df.head(20))