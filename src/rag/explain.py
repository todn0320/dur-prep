def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def shorten_text(text, max_len=120):
    text = safe_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def generate_explanation(drug_info: dict) -> str:
    selected_item = drug_info.get("selected_item")
    candidates = drug_info.get("candidates", [])

    if not selected_item or not candidates:
        return "약 정보를 확인하지 못했습니다."

    top = candidates[0]

    item_name = safe_text(top.get("item_name"))
    entp_name = safe_text(top.get("entp_name"))
    etc_otc_code = safe_text(top.get("etc_otc_code"))
    effect = shorten_text(top.get("effect"))
    usage = shorten_text(top.get("usage"))
    warning = shorten_text(top.get("warning"))
    interaction = shorten_text(top.get("interaction"))

    parts = []

    if item_name:
        parts.append(f"인식된 약 후보 중 가장 유력한 약은 {item_name}입니다.")

    if entp_name:
        parts.append(f"제조사는 {entp_name}입니다.")

    if etc_otc_code:
        parts.append(f"의약품 구분은 {etc_otc_code}입니다.")

    if effect:
        parts.append(f"효능·효과: {effect}")

    if usage:
        parts.append(f"용법·용량: {usage}")

    if warning:
        parts.append(f"주의사항: {warning}")

    if interaction:
        parts.append(f"상호작용: {interaction}")

    return " ".join(parts)