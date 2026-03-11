import os
import cv2
import tempfile
import easyocr

from src.ocr.normalize import normalize_imprint
from src.ocr.preprocess_ocr import generate_ocr_variants


reader = easyocr.Reader(['en'], gpu=False)


def _read_with_easyocr(image_input):
    return reader.readtext(
        image_input,
        detail=0,
        paragraph=False,
        batch_size=1,
        canvas_size=1280,
        mag_ratio=1.0
    )


def _deduplicate_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def run_ocr(image_path: str):
    all_raw = []
    all_norm = []

    variants = generate_ocr_variants(image_path)

    for variant_name, variant_img in variants:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, variant_img)

            results = _read_with_easyocr(temp_path)

            for text in results:
                raw_text = str(text).strip()
                norm_text = normalize_imprint(raw_text)

                if raw_text:
                    all_raw.append(raw_text)
                if norm_text:
                    all_norm.append(norm_text)

        except Exception as e:
            print(f"[OCR] {variant_name} 실패: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    all_raw = _deduplicate_keep_order(all_raw)
    all_norm = _deduplicate_keep_order(all_norm)

    return {
        "ocr_raw": all_raw,
        "ocr_norm": all_norm
    }