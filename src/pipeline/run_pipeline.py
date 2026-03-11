import os
import json

from src.inference.predictor import predict_topk
from src.ocr.ocr_engine import run_ocr
from src.db.query_drug import query_drug
from src.rag.explain import generate_explanation


def run_pipeline(image_path: str):
    topk = predict_topk(image_path, k=5)
    ocr_result = run_ocr(image_path)
    drug_info = query_drug(topk, ocr_result)
    rag_text = generate_explanation(drug_info)

    result = {
        "topk": topk,
        "ocr": ocr_result,
        "drug_info": drug_info,
        "rag_text": rag_text
    }
    return result


if __name__ == "__main__":
    image_path = "release/demo_samples/sample.png"

    if not os.path.exists(image_path):
        print("이미지 파일을 찾을 수 없습니다.")
    else:
        result = run_pipeline(image_path)
        print("\n===== PIPELINE RESULT =====")
        print(json.dumps(result, ensure_ascii=False, indent=2))