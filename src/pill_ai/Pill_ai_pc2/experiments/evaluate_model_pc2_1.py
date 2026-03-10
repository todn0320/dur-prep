from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


# =========================
# 1. 경로 설정
# =========================
META_CSV = Path("src/pill_ai/Pill_ai_pc2/data/meta/meta_split.csv")
MODEL_PATH = Path("src/pill_ai/Pill_ai_pc2/models/resnet18_baseline_pc2_1.pt")
LABEL_MAP_PATH = Path("src/pill_ai/Pill_ai_pc2/models/label_map_pc2_1.json")
OUTPUT_DIR = Path("src/pill_ai/Pill_ai_pc2/models/eval_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CM_TXT_PATH = OUTPUT_DIR / "confusion_matrix.csv"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
PRED_PATH = OUTPUT_DIR / "val_predictions.csv"
CM_FIG_PATH = OUTPUT_DIR / "confusion_matrix.png"


# =========================
# 2. 설정
# =========================
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0


# =========================
# 3. 디바이스
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")


# =========================
# 4. 입력 파일 체크
# =========================
if not META_CSV.exists():
    raise FileNotFoundError(f"meta_split.csv 없음: {META_CSV}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"학습된 모델 없음: {MODEL_PATH}")

if not LABEL_MAP_PATH.exists():
    raise FileNotFoundError(f"label_map 없음: {LABEL_MAP_PATH}")


# =========================
# 5. 데이터 읽기
# =========================
df = pd.read_csv(META_CSV)
val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

class_to_idx = label_map["class_to_idx"]
idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}

num_classes = len(class_to_idx)

print(f"val 개수: {len(val_df)}")
print(f"클래스 수: {num_classes}")


# =========================
# 6. Transform
# =========================
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# =========================
# 7. Dataset
# =========================
class PillDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, class_to_idx: dict, transform=None):
        self.df = data_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["image_path"])
        class_id = str(row["class_id"])

        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        label = self.class_to_idx[class_id]

        if self.transform:
            image = self.transform(image)

        return image, label, row["image_path"], class_id


val_dataset = PillDataset(val_df, class_to_idx, transform=val_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================
# 8. 모델 로드
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# =========================
# 9. 평가
# =========================
all_preds = []
all_labels = []
all_paths = []
all_true_class = []
all_pred_class = []

with torch.no_grad():
    for images, labels, paths, true_class_ids in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        preds_cpu = preds.cpu().numpy().tolist()
        labels_cpu = labels.cpu().numpy().tolist()

        all_preds.extend(preds_cpu)
        all_labels.extend(labels_cpu)
        all_paths.extend(list(paths))
        all_true_class.extend(list(true_class_ids))
        all_pred_class.extend([idx_to_class[p] for p in preds_cpu])

acc = accuracy_score(all_labels, all_preds)
print(f"\nValidation Accuracy: {acc:.4f}")


# =========================
# 10. classification report
# =========================
target_names = [idx_to_class[i] for i in range(num_classes)]
report = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    digits=4,
    zero_division=0
)

print("\n=== Classification Report ===")
print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy: {acc:.4f}\n\n")
    f.write(report)


# =========================
# 11. confusion matrix 저장
# =========================
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_df.to_csv(CM_TXT_PATH, encoding="utf-8-sig")


# =========================
# 12. 예측 결과 저장
# =========================
pred_df = pd.DataFrame({
    "image_path": all_paths,
    "true_class_id": all_true_class,
    "pred_class_id": all_pred_class,
    "is_correct": [t == p for t, p in zip(all_true_class, all_pred_class)]
})
pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")


# =========================
# 13. confusion matrix 시각화
# =========================
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(num_classes), target_names, rotation=90)
plt.yticks(range(num_classes), target_names)
plt.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(CM_FIG_PATH, dpi=200, bbox_inches="tight")
plt.close()


print("\n=== 저장 완료 ===")
print(f"report: {REPORT_PATH}")
print(f"confusion_matrix csv: {CM_TXT_PATH}")
print(f"confusion_matrix png: {CM_FIG_PATH}")
print(f"predictions: {PRED_PATH}")