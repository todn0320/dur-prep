import os

from src.inference.predictor import predict_topk
from src.ocr.ocr_engine import run_ocr
from src.db.query_drug import query_drug

def run_pipeline(image_path: str):
    topk = predict_topk(image_path, k=5)
    ocr_result = run_ocr(image_path)
    drug_info = query_drug(topk, ocr_result)

    result = {
        "topk": topk,
        "ocr": ocr_result,
        "drug_info": drug_info,
        "rag_text": f"{drug_info['item_name']}의 정보를 안내합니다."
    }
    return result

if __name__ == "__main__":
    image_path = "release/demo_samples/sample.png"

    if not os.path.exists(image_path):
        print("이미지 파일을 찾을 수 없습니다.")
    else:
        result = run_pipeline(image_path)
        print("\n===== PIPELINE RESULT =====")
        print(result)