import os
import json
import torch

from src.inference.model_loader import load_model
from src.inference.preprocess import preprocess_image


MODEL_PATH = os.path.join("release", "models", "pill_cls_best.pt.pt")
LABEL_MAP_PATH = os.path.join("release", "models", "label_map_pc.json")


def load_label_map(label_map_path: str):
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_to_item_seq(label_map: dict):
    """
    다양한 label_map 구조 대응

    지원 형태:
    1) {"1111": 0, "2222": 1}
    2) {"0": "1111", "1": "2222"}
    3) {
         "class_to_idx": {"1111": 0, "2222": 1},
         "idx_to_class": {"0": "1111", "1": "2222"}
       }
    """
    class_to_item_seq = {}

    if not label_map:
        return class_to_item_seq

    if "idx_to_class" in label_map:
        idx_to_class = label_map["idx_to_class"]
        return {int(k): str(v) for k, v in idx_to_class.items()}

    if "class_to_idx" in label_map:
        class_to_idx = label_map["class_to_idx"]
        return {int(v): str(k) for k, v in class_to_idx.items()}

    sample_key = next(iter(label_map.keys()))
    sample_val = label_map[sample_key]

    if isinstance(sample_val, int):
        for item_seq, class_idx in label_map.items():
            class_to_item_seq[int(class_idx)] = str(item_seq)
    else:
        for class_idx, item_seq in label_map.items():
            class_to_item_seq[int(class_idx)] = str(item_seq)

    return class_to_item_seq


def predict_topk(image_path: str, k: int = 5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"입력 이미지가 없습니다: {image_path}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"label_map 파일이 없습니다: {LABEL_MAP_PATH}")

    label_map = load_label_map(LABEL_MAP_PATH)
    class_to_item_seq = build_class_to_item_seq(label_map)
    num_classes = len(class_to_item_seq)

    if num_classes == 0:
        raise ValueError("label_map에서 클래스 정보를 읽지 못했습니다.")

    model, device = load_model(MODEL_PATH, num_classes=num_classes)
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(k, num_classes), dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    results = []
    for idx, score in zip(top_indices, top_probs):
        item_seq = class_to_item_seq.get(idx, f"unknown_{idx}")
        results.append({
            "item_seq": item_seq,
            "score": round(float(score), 4)
        })

    return results