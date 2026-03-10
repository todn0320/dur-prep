from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

META_CSV = Path("src/pill_ai/Pill_ai_pc4/data/meta/meta.csv")
OUT_CSV = Path("src/pill_ai/Pill_ai_pc4/data/meta/meta_split.csv")

df = pd.read_csv(META_CSV)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["class_id"],
    random_state=42
)

train_df = train_df.copy()
val_df = val_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"

out_df = pd.concat([train_df, val_df], ignore_index=True)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("저장 완료:", OUT_CSV)
print(out_df["split"].value_counts())
print("\n클래스별 split 분포:")
print(out_df.groupby(["class_id", "split"]).size().unstack(fill_value=0))