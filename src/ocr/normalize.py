def normalize_imprint(text: str) -> str:
    if not text:
        return ""

    text = text.upper()
    text = text.replace("-", "")
    text = text.replace(" ", "")
    text = text.replace("_", "")
    return text