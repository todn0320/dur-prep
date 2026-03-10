import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ==========================================
# 1. 경로 설정 (PC3 기준)
# ==========================================
BASE_DIR = Path("src/pill_ai/Pill_ai_pc3")
CSV_PATH = BASE_DIR / "data/meta/meta_split.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODEL_DIR / "resnet18_aug_pc3_1.pt"

# 저장할 모델 폴더가 없으면 자동 생성
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 커스텀 Dataset 클래스
# ==========================================
class PillDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 이미지가 RGBA 형태일 수 있으므로 RGB로 강제 변환
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # ==========================================
    # 3. 데이터 준비 및 라벨 인코딩
    # ==========================================
    print("⏳ 데이터를 불러오는 중입니다...")
    df = pd.read_csv(CSV_PATH)

    # class_id를 PyTorch가 이해할 수 있는 정수 라벨(0~21)로 매핑
    unique_classes = sorted(df["class_id"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    df["label"] = df["class_id"].map(class_to_idx)

    # Train / Val 분리
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"✅ Train 데이터: {len(train_df)}장")
    print(f"✅ Val 데이터: {len(val_df)}장")
    print(f"✅ 총 분류 클래스 수: {len(unique_classes)}개\n")

    # ==========================================
    # 4. PC3 전용 Augmentation 전략 (Shape Robustness)
    # ==========================================
    # Input size 224를 맞추기 위해, 먼저 256으로 늘린 후 224로 RandomCrop 진행
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ⚠️ Validation은 성능의 정확한 측정을 위해 Augmentation을 적용하지 않음
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PillDataset(train_df, transform=train_transform)
    val_dataset = PillDataset(val_df, transform=val_transform)

    # num_workers는 윈도우 환경을 고려해 0으로 설정 (속도를 높이려면 4 등으로 변경 가능)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ==========================================
    # 5. 모델 세팅 (ResNet18 Pretrained)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 사용 기기: {device}")

    # 최신 PyTorch 버전에 맞춘 사전학습 가중치 로드 방식
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 마지막 FC(Fully Connected) 레이어를 알약 클래스 수(22)에 맞게 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(unique_classes))
    model = model.to(device)

    # ==========================================
    # 6. Loss 및 Optimizer (Adam) 설정
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    # ==========================================
    # 7. 학습 루프 (Train & Validation)
    # ==========================================
    print("🔥 모델 학습을 시작합니다...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = 100.0 * correct / total

        # Validation 평가
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # ==========================================
    # 8. 학습 완료 후 모델 저장
    # ==========================================
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎉 학습 완료! 모델이 성공적으로 저장되었습니다: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
