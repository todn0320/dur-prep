import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
# =========================
# 1. 경로 설정 (PC5 학습 코드와 동기화)
# =========================
BASE_DIR = Path("src/pill_ai/Pill_ai_pc5")
CSV_PATH = BASE_DIR / "data/meta/meta_split.csv"
MODEL_PATH = BASE_DIR / "models/resnet18_aug_pc5_1.pt"

# 결과 저장 경로
OUTPUT_DIR = BASE_DIR / "models/eval_results_aug_pc5_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "classification_report_aug_pc5_1.txt"
CM_TXT_PATH = OUTPUT_DIR / "confusion_matrix_aug_pc5_1.csv"
CM_FIG_PATH = OUTPUT_DIR / "confusion_matrix_aug_pc5_1.png"
PRED_PATH = OUTPUT_DIR / "val_predictions_aug_pc5_1.csv"

# =========================
# 2. 데이터셋 클래스 (학습 코드와 동일)
# =========================
class PillDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_paths = df['image_path'].tolist()
        self.class_ids = df['class_id'].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        class_id = self.class_ids[idx]
        label = self.class_to_idx[class_id]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path, class_id

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 시작 (Device: {device})")

    # =========================
    # 3. 데이터 준비 및 라벨 매핑
    # =========================
    df = pd.read_csv(CSV_PATH)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # 학습 시와 동일한 라벨 순서 보장
    unique_classes = sorted(df['class_id'].unique().astype(str))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(unique_classes)

    # Validation Transform (Heavy Augmentation 미적용, Baseline과 동일)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = PillDataset(val_df, class_to_idx, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # =========================
    # 4. 모델 로드
    # =========================
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # state_dict 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # =========================
    # 5. 추론 및 평가
    # =========================
    all_preds = []
    all_labels = []
    all_paths = []
    all_true_class_ids = []
    all_pred_class_ids = []

    print("🧪 추론 진행 중...")
    with torch.no_grad():
        for images, labels, paths, class_ids in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            all_true_class_ids.extend(class_ids)
            all_pred_class_ids.extend([idx_to_class[p] for p in preds.cpu().numpy()])

    # 정확도 계산
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✨ Validation Accuracy: {acc:.4f}")

    # =========================
    # 6. 결과 저장 (Report, CM, CSV)
    # =========================
    # 1) Classification Report
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\n=== Classification Report ===")
    print(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 2) Confusion Matrix CSV
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(CM_TXT_PATH, encoding="utf-8-sig")

    # 3) Confusion Matrix 시각화
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
    plt.savefig(OUTPUT_DIR / "confusion_matrix_aug_pc5_1.png")
    plt.show()

    # 4) 예측 상세 결과 CSV
    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "true_class_id": all_true_class_ids,
        "pred_class_id": all_pred_class_ids,
        "is_correct": [t == p for t, p in zip(all_true_class_ids, all_pred_class_ids)]
    })
    pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 모든 평가 결과가 저장되었습니다: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()