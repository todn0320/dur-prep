import easyocr
import cv2
import tempfile
import os

from src.ocr.normalize import normalize_imprint
from src.ocr.preprocess_ocr import preprocess_for_ocr

reader = easyocr.Reader(['en'], gpu=False)


def _read_with_easyocr(image_input):
    results = reader.readtext(image_input, detail=0)

    ocr_raw = []
    ocr_norm = []

    for text in results:
        raw_text = str(text).strip()
        norm_text = normalize_imprint(raw_text)

        if raw_text:
            ocr_raw.append(raw_text)
        if norm_text:
            ocr_norm.append(norm_text)

    return ocr_raw, ocr_norm


def run_ocr(image_path: str):
    all_raw = []
    all_norm = []

    # 1) 원본 OCR
    raw1, norm1 = _read_with_easyocr(image_path)
    all_raw.extend(raw1)
    all_norm.extend(norm1)

    # 2) 전처리본 OCR
    processed = preprocess_for_ocr(image_path)
    if processed is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, processed)

        raw2, norm2 = _read_with_easyocr(temp_path)
        all_raw.extend(raw2)
        all_norm.extend(norm2)

        os.remove(temp_path)

    # 중복 제거
    all_raw = list(dict.fromkeys(all_raw))
    all_norm = list(dict.fromkeys(all_norm))

    return {
        "ocr_raw": all_raw,
        "ocr_norm": all_norm
    }