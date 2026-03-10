import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path

# ==============================
# 1. 경로 및 하이퍼파라미터 설정
# ==============================
# 현재 스크립트 위치: src/pill_ai/Pill_ai_pc2/experiments/
BASE_DIR = Path(__file__).resolve().parent.parent
# 최상위 루트 폴더 (dur-prep)
ROOT_DIR = BASE_DIR.parent.parent.parent

CSV_PATH = BASE_DIR / "data" / "meta" / "meta_split.csv"
SAVE_DIR = BASE_DIR / "models"
SAVE_PATH = SAVE_DIR / "resnet18_aug_pc2_1.pt"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INPUT_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# ==============================
# 2. PC2 전용 데이터 증강 전략 (Augmentation)
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================
# 3. 데이터셋 클래스 (경로 에러 및 라벨 매핑 완벽 반영)
# ==============================
class PillDataset(Dataset):
    def __init__(self, dataframe, root_dir, class_to_idx, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # 팩트: 최상위 폴더(dur-prep) 기준으로 경로 연결 (경로 중복 방지)
        img_path = str(self.root_dir / row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # 팩트: 원본 class_id (예: 41170)를 파이토치용 인덱스 (0, 1...)로 변환
        label = self.class_to_idx[row['class_id']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==============================
# 4. 데이터 로드 및 분할
# ==============================
print("메타 데이터(meta_split.csv) 불러오는 중...")
df = pd.read_csv(CSV_PATH)

# 라벨 매핑 딕셔너리 생성
unique_classes = sorted(df['class_id'].unique())
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
num_classes = len(unique_classes)

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)

print(f"총 클래스 수: {num_classes} / Train 데이터: {len(train_df)}장 / Val 데이터: {len(val_df)}장")

train_dataset = PillDataset(train_df, ROOT_DIR, class_to_idx, transform=train_transform)
val_dataset = PillDataset(val_df, ROOT_DIR, class_to_idx, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# 5. 모델 정의 (ResNet18)
# ==============================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==============================
# 6. 학습 루프 및 정확한 출력 포맷
# ==============================
print(f"학습 시작: 총 {EPOCHS} 에포크")

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    train_loss = train_running_loss / len(train_loader)
    train_acc = train_correct / train_total  # 0~1 소수점 유지
    
    # --- Validation ---
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total  # 0~1 소수점 유지
    
    # 요구하신 완벽한 출력 포맷 팩트 체크
    print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# ==============================
# 7. 최종 모델 저장
# ==============================
torch.save(model.state_dict(), SAVE_PATH)
print(f"학습 완료! PC2 모델 저장 됨: {SAVE_PATH}")