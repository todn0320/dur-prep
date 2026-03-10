from pathlib import Path
import json
import copy
import random

import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score


# =========================
# 1. 경로 설정
# =========================
META_CSV = Path("src/pill_ai/Pill_ai_pc1/data/meta/meta_split.csv")
MODEL_DIR = Path("src/pill_ai/Pill_ai_pc1/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_baseline_pc1_1.pt"
LABEL_MAP_PATH = MODEL_DIR / "label_map_pc1_1.json"


# =========================
# 2. 하이퍼파라미터
# =========================
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # 윈도우면 0 추천


# =========================
# 3. 시드 고정
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# 4. 디바이스
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")


# =========================
# 5. 데이터 읽기
# =========================
if not META_CSV.exists():
    raise FileNotFoundError(f"meta_split.csv 파일이 없습니다: {META_CSV}")

df = pd.read_csv(META_CSV)

required_cols = {"image_path", "class_id", "split"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

print(f"train 개수: {len(train_df)}")
print(f"val 개수: {len(val_df)}")


# =========================
# 6. 라벨 인코딩
# =========================
class_ids = sorted(df["class_id"].astype(str).unique().tolist())
class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
idx_to_class = {idx: class_id for class_id, idx in class_to_idx.items()}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"클래스 수: {len(class_ids)}")
print(f"라벨맵 저장 완료: {LABEL_MAP_PATH}")


# =========================
# 7. Transform
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# =========================
# 8. Dataset
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

        return image, label


train_dataset = PillDataset(train_df, class_to_idx, transform=train_transform)
val_dataset = PillDataset(val_df, class_to_idx, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================
# 9. 모델 정의
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_ids))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================
# 10. 학습/평가 함수
# =========================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


# =========================
# 11. 학습 루프
# =========================
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    print(
        f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())


# =========================
# 12. 최고 성능 모델 저장
# =========================
model.load_state_dict(best_model_wts)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": IMAGE_SIZE,
    },
    MODEL_PATH
)

print(f"\n최고 val_acc: {best_val_acc:.4f}")
print(f"모델 저장 완료: {MODEL_PATH}")