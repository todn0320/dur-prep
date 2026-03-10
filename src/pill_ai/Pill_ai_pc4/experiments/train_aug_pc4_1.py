import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 1. 경로 설정
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_file_path))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# 2. Dataset 클래스 (Label Encoding 추가)
class PillDataset(Dataset):
    def __init__(self, csv_file, split_type, transform=None):
        full_df = pd.read_csv(csv_file)
        
        # 0~21 인덱스 매핑 생성 (IndexError 해결 핵심)
        self.unique_labels = sorted(full_df['class_id'].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        
        # 해당 split 데이터만 필터링
        self.df = full_df[full_df['split'] == split_type].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.df.loc[idx, 'image_path'].replace('\\', '/')
        
        # 경로 중복 해결
        if 'data/sample_img' in raw_path:
            target_sub_path = raw_path.split('data/sample_img')[-1].lstrip('/')
            img_path = os.path.join(PROJECT_ROOT, 'data', 'sample_img', target_sub_path)
        else:
            img_path = os.path.join(PROJECT_ROOT, raw_path)

        image = Image.open(img_path).convert('RGB')
        
        # 실제 ID(ex: 41097)를 0~21 범위의 인덱스로 변환
        label_raw = self.df.loc[idx, 'class_id']
        label = self.label_to_idx[label_raw]

        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    CSV_PATH = os.path.join(PROJECT_ROOT, "data", "meta", "meta_split.csv")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
    SAVE_PATH = os.path.join(SAVE_DIR, "resnet18_aug_pc4_1.pt")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    NUM_CLASSES = 22
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 시작 (Device: {device}, Classes: {NUM_CLASSES})...")

    # 3. Augmentation (PC4: Robustness 실험)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 0.5))], p=0.3),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.02),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = PillDataset(CSV_PATH, 'train', transform=train_transform)
    val_dataset = PillDataset(CSV_PATH, 'val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 모델 설정
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_acc = correct_train / total_train
        train_loss = running_loss / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_acc = correct_val / total_val
        val_loss /= total_val

        # 출력 포맷 준수
        print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"모델 저장 완료: {SAVE_PATH}")

if __name__ == '__main__':
    main()