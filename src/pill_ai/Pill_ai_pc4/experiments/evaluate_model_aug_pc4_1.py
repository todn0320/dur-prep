from pathlib import Path
import json
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 경로 설정
# =========================
# 파일 위치에 상관없이 프로젝트 루트를 정확히 잡도록 절대 경로 활용
current_file_path = Path(__file__).resolve()
# 보통 experiments 폴더 안에 스크립트가 있으므로 부모의 부모 폴더가 프로젝트 루트임
PROJECT_ROOT = current_file_path.parent.parent 

META_CSV = PROJECT_ROOT / "data" / "meta" / "meta_split.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_aug_pc4_1.pt"

RESULT_DIR = PROJECT_ROOT / "models" / "eval_results_aug"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# device 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)


# =========================
# 메타 로드 및 매핑 생성
# =========================
df = pd.read_csv(META_CSV)
val_df = df[df["split"] == "val"].reset_index(drop=True)
print("val 개수:", len(val_df))

# 학습 시와 동일하게 고유 class_id를 추출하여 매핑 정보를 직접 생성
unique_labels = sorted(df["class_id"].unique())
class_to_idx = {str(label): i for i, label in enumerate(unique_labels)}
idx_to_class = {i: str(label) for i, label in enumerate(unique_labels)}
num_classes = len(unique_labels)


# =========================
# 모델 로드
# =========================
# checkpoint에는 가중치(state_dict)만 들어있음
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 딕셔너리가 아닌 가중치 자체를 로드
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


# =========================
# transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# =========================
# dataset (경로 오류 방지 로직)
# =========================
class PillDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = str(row["image_path"]).replace('\\', '/')

        # 경로 중복 해결을 위해 sample_img 이후 경로만 추출하여 결합
        if 'data/sample_img' in raw_path:
            target_sub_path = raw_path.split('data/sample_img')[-1].lstrip('/')
            img_path = PROJECT_ROOT / "data" / "sample_img" / target_sub_path
        else:
            img_path = PROJECT_ROOT / raw_path

        label = class_to_idx[str(row["class_id"])]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, row["class_id"], str(img_path)

dataset = PillDataset(val_df, transform)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


# =========================
# inference
# =========================
all_preds = []
all_labels = []
paths = []

with torch.no_grad():
    for imgs, labels, class_ids, img_paths in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        paths.extend(img_paths)


# =========================
# classification report
# =========================
target_names = [idx_to_class[i] for i in range(num_classes)]

report = classification_report(
    all_labels,
    all_preds,
    target_names=target_names
)

print(report)

with open(RESULT_DIR / "classification_report_aug_pc4_1.txt", "w", encoding="utf-8") as f:
    f.write(report)


# =========================
# confusion matrix
# =========================
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_df.to_csv(RESULT_DIR / "confusion_matrix_aug_pc4_1.csv")

plt.figure(figsize=(10,8))
sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="viridis"
)
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix_aug_pc4_1.png")
plt.show()


# =========================
# prediction 저장 (is_correct 컬럼 추가)
# =========================
pred_class = [idx_to_class[p] for p in all_preds]
true_class = [idx_to_class[t] for t in all_labels]
is_correct = [t == p for t, p in zip(true_class, pred_class)]

pred_df = pd.DataFrame({
    "image_path": paths,
    "true": true_class,
    "pred": pred_class,
    "is_correct": is_correct
})

pred_df.to_csv(RESULT_DIR / "val_predictions_aug_pc4_1.csv", index=False)

print("\n결과 저장 완료:")
print(RESULT_DIR)