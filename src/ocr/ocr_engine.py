import easyocr
import cv2
import tempfile
import os

from src.ocr.preprocess_ocr import generate_ocr_variants
from src.ocr.normalize import normalize_imprint

reader = easyocr.Reader(['en'], gpu=False)


def run_ocr(image_path: str):

    variants = generate_ocr_variants(image_path)

    ocr_raw = []
    ocr_norm = []

    for variant_name, variant_img in variants:

        temp_path = None

        try:
            # OpenCV 이미지를 OCR이 읽을 수 있도록 임시 파일 저장
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, variant_img)

            # EasyOCR 실행
            results = reader.readtext(temp_path, detail=0)

            # 결과 누적
            for text in results:

                raw = str(text).strip()
                norm = normalize_imprint(raw)

                if raw:
                    ocr_raw.append(raw)

                if norm:
                    ocr_norm.append(norm)

        except Exception as e:
            print(f"OCR 실패 ({variant_name}):", e)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # 중복 제거
    ocr_raw = list(dict.fromkeys(ocr_raw))
    ocr_norm = list(dict.fromkeys(ocr_norm))

    return {
        "ocr_raw": ocr_raw,
        "ocr_norm": ocr_norm
    }