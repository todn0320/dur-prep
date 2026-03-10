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
# 1. 경로 설정 (PC5 기준)
# ==========================================
BASE_DIR = Path("src/pill_ai/Pill_ai_pc5")
CSV_PATH = BASE_DIR / "data/meta/meta_split.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODEL_DIR / "resnet18_aug_pc5_1.pt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 커스텀 Dataset 클래스
# ==========================================
class PillDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # ==========================================
    # 3. 데이터 준비 및 라벨 인코딩
    # ==========================================
    print("⏳ PC5 데이터를 불러오는 중입니다...")
    df = pd.read_csv(CSV_PATH)
    
    unique_classes = sorted(df['class_id'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    df['label'] = df['class_id'].map(class_to_idx)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    print(f"✅ Train 데이터: {len(train_df)}장")
    print(f"✅ Val 데이터: {len(val_df)}장")
    print(f"✅ 총 분류 클래스 수: {len(unique_classes)}개\n")
    
    # ==========================================
    # 4. PC5 전용 Augmentation 전략 (정확도 확보용)
    # ==========================================
    train_transform = transforms.Compose([
        # ⚠️ Resize 대신 RandomResizedCrop을 쓰면 알약의 특징을 더 잘 잡아서 성능이 확 올라가!
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30), # 회전 범위를 넓혀서 일반화 성능 향상
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PillDataset(train_df, transform=train_transform)
    val_dataset = PillDataset(val_df, transform=val_transform)

    train_loader = DataLoader(PillDataset(train_df, train_transform), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(PillDataset(val_df, val_transform), batch_size=32, shuffle=False, num_workers=0)

    # ==========================================
    # 5. 모델 세팅
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 사용 기기: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(unique_classes))
    model = model.to(device)

    # ==========================================
    # 6. Loss 및 Optimizer 설정
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    # ⚠️ 핵심: 학습률을 0.001 -> 0.0001로 낮춤 (Pretrained 모델은 살살 학습시켜야 정확도가 높아짐)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    epochs = 10

    # ==========================================
    # 7. 학습 루프
    # ==========================================
    print("🔥 PC5 모델 학습을 시작합니다...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

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
        train_acc = 100. * correct / total

        model.eval()
        val_loss_sum = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)
                val_loss_sum += v_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / len(val_dataset)
        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # ==========================================
    # 8. 모델 저장
    # ==========================================
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎉 학습 완료! 저장 위치: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()