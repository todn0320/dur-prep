import os
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # ✅ seaborn 추가

# =========================
# 1. 경로 설정 (PC3 Augmentation 모델 기준)
# =========================
BASE_DIR = Path("src/pill_ai/Pill_ai_pc3")
META_CSV = BASE_DIR / "data/meta/meta_split.csv"
MODEL_PATH = BASE_DIR / "models/resnet18_aug_pc3_1.pt"

# 결과를 저장할 폴더 (baseline과 섞이지 않도록 별도 폴더 권장)
OUTPUT_DIR = BASE_DIR / "experiments/eval_results_aug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4개의 결과물 출력 경로 설정
REPORT_PATH = OUTPUT_DIR / "classification_report_aug_pc3_1.txt"
CM_TXT_PATH = OUTPUT_DIR / "confusion_matrix_aug_pc3_1.csv"
PRED_PATH = OUTPUT_DIR / "val_predictions_aug_pc3_1.csv"
CM_FIG_PATH = OUTPUT_DIR / "confusion_matrix_aug_pc3_1.png"

# =========================
# 2. 설정 및 디바이스
# =========================
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 사용 디바이스: {device}")

# =========================
# 3. 데이터 읽기 및 라벨 자동 매핑 (json 불필요)
# =========================
if not META_CSV.exists():
    raise FileNotFoundError(f"meta_split.csv 없음: {META_CSV}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"학습된 모델 없음: {MODEL_PATH}")

df = pd.read_csv(META_CSV)

# meta.csv의 class_id를 알파벳/숫자 순으로 정렬하여 0~21 인덱스 자동 부여
unique_classes = sorted(df['class_id'].astype(str).unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
num_classes = len(unique_classes)

val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

print(f"✅ 평가할 Validation 데이터 수: {len(val_df)}")
print(f"✅ 총 클래스 수: {num_classes}")

# =========================
# 4. Transform (학습 시의 Validation 세팅과 동일)
# =========================
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# 5. Dataset & DataLoader
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
        image = ImageOps.exif_transpose(image) # 이미지 회전 메타데이터 자동 교정
        
        label = self.class_to_idx[class_id]

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path), class_id

val_dataset = PillDataset(val_df, class_to_idx, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# =========================
# 6. 모델 로드
# =========================
print(f"⏳ 모델 가중치 불러오는 중... ({MODEL_PATH.name})")
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

# 우리가 저장했던 모델 가중치 불러오기
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# 7. 평가 (추론 진행)
# =========================
all_preds = []
all_labels = []
all_paths = []
all_true_class = []

print("🔥 추론 시작...")
with torch.no_grad():
    for images, labels, paths, true_class_ids in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_paths.extend(list(paths))
        all_true_class.extend(list(true_class_ids))

all_pred_class = [idx_to_class[p] for p in all_preds]

# =========================
# 8. 결과 계산 및 리포트 저장 (4가지 출력물)
# =========================
# [1] 전체 정확도 및 Classification Report (.txt)
acc = accuracy_score(all_labels, all_preds)
target_names = [idx_to_class[i] for i in range(num_classes)]

report = classification_report(
    all_labels, all_preds, 
    target_names=target_names, 
    digits=4, zero_division=0
)

print(f"\n🎉 Validation Accuracy: {acc * 100:.2f}%")
print("\n=== Classification Report ===")
print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy: {acc:.4f}\n\n")
    f.write(report)

# [2] Confusion Matrix (.csv)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_df.to_csv(CM_TXT_PATH, encoding="utf-8-sig")

# [3] 예측 결과 매핑 (.csv)
pred_df = pd.DataFrame({
    "image_path": all_paths,
    "true_class_id": all_true_class,
    "pred_class_id": all_pred_class,
    "is_correct": [t == p for t, p in zip(all_true_class, all_pred_class)]
})
pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

# =========================
# [4] Confusion Matrix 시각화 (.png) ✅ 수정 완료
# =========================
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="viridis"
)
plt.title("Confusion Matrix (Shape Augmentation - PC3)")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.savefig(CM_FIG_PATH, dpi=200, bbox_inches="tight")
plt.show()  # 화면에 띄우기
plt.close() # 메모리 확보를 위해 닫기

print("\n=== 💾 결과물 저장 완료 ===")
print(f"1. Report txt : {REPORT_PATH}")
print(f"2. CM csv     : {CM_TXT_PATH}")
print(f"3. Preds csv  : {PRED_PATH}")
print(f"4. CM png     : {CM_FIG_PATH}")