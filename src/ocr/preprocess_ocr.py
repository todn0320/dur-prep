import cv2


def resize_if_needed(image, max_side=1280):
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def generate_ocr_variants(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return []

    image = resize_if_needed(image, max_side=1280)

    variants = []

    # 1) 원본
    variants.append(("original", image))

    # 2) gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    # 3) adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    variants.append(("adaptive", thresh))

    # 4) 확대본
    enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("enlarged_gray", enlarged))

    # 5) 확대 + threshold
    enlarged_thresh = cv2.adaptiveThreshold(
        enlarged, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    variants.append(("enlarged_adaptive", enlarged_thresh))

    return variants

# threshold 추가됨
# 나중에 여기에 그레이스케일,crop 추가