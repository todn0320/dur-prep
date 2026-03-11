import cv2


def preprocess_for_ocr(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 대비 강화용 threshold
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    return thresh

# 그레이스케일 + threshold 추가됨
# 나중에 여기에 crop 추가