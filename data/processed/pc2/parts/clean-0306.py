from pathlib import Path
import pandas as pd
import json
import re
from typing import Optional

# =========================
# 1) 경로 설정 (프로젝트 루트 기준)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "processed" / "pc2" / "clean_by_type"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pc2" / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2) 공통 유틸
# =========================

NONE_LIKE = {"None", "none", "NULL", "null", "nan", "NaN", "", " ", ".", "<NA>", "-"}

DATE_COL_HINTS = {
    "notification_date", "item_permit_date", "change_date",
    "open_date", "update_date", "cancel_date", "img_regist_date",
    "opende", "updatede", "img_regist_ts", "item_permit_date"
}

NUMERIC_COLS_HINTS = {
    "warning_type", "leng_long", "leng_short", "thick",
    "class_no", "qnt", "entp_seq"
}

COLUMN_RENAME_MAP = {
    # 공통
    "ITEM_SEQ": "item_seq",
    "ITEM_NAME": "item_name",
    "ENTP_NAME": "entp_name",
    "ITEM_PERMIT_DATE": "item_permit_date",
    "CHANGE_DATE": "change_date",
    "ATC_CODE": "atc_code",
    "EDI_CODE": "edi_code",
    "BIZRNO": "bizrno",
    "ITEM_ENG_NAME": "item_eng_name",
    "ENTP_SEQ": "entp_seq",

    # DUR
    "TYPE_NAME": "type_name",
    "INGR_CODE": "ingr_code",
    "MIXTURE_INGR_CODE": "mixture_ingr_code",

    # 주성분
    "MTRAL_CODE": "mtral_code",
    "MTRAL_NM": "mtral_nm",
    "QNT": "qnt",
    "INGD_UNIT_CD": "ingd_unit_cd",
    "MAIN_INGR_ENG": "main_ingr_eng",

    # 허가목록
    "SPCLTY_PBLC": "etc_otc",
    "PRDUCT_TYPE": "product_type",
    "ITEM_INGR_NAME": "main_ingr_eng",
    "CANCEL_DATE": "cancel_date",

    # 허가상세
    "ETC_OTC_CODE": "etc_otc_code",
    "EE_DOC_DATA": "ee_doc_data",
    "UD_DOC_DATA": "ud_doc_data",
    "NB_DOC_DATA": "nb_doc_data",
    "MAIN_ITEM_INGR": "main_item_ingr",
    "INGR_NAME": "ingr_name",

    # 낱알식별
    "CHART": "chart",
    "ITEM_IMAGE": "item_image_url",
    "PRINT_FRONT": "print_front",
    "PRINT_BACK": "print_back",
    "DRUG_SHAPE": "drug_shape",
    "COLOR_CLASS1": "color_class1",
    "COLOR_CLASS2": "color_class2",
    "LINE_FRONT": "line_front",
    "LINE_BACK": "line_back",
    "LENG_LONG": "leng_long",
    "LENG_SHORT": "leng_short",
    "THICK": "thick",
    "IMG_REGIST_TS": "img_regist_date",
    "CLASS_NO": "class_no",
    "CLASS_NAME": "class_name",
    "ETC_OTC_NAME": "etc_otc_name",
    "FORM_CODE_NAME": "form_code_name",
    "MARK_CODE_FRONT": "mark_code_front",
    "MARK_CODE_BACK": "mark_code_back",
    "STD_CD": "std_cd",

    # e약은요
    "entpname": "entp_name",
    "itemname": "item_name",
    "itemseq": "item_seq",
    "efcyqesitm": "efficacy_text_easy",
    "usemethodqesitm": "usage_text_easy",
    "atpnwarnqesitm": "warning_text_easy",
    "atpnqesitm": "caution_text_easy",
    "intrcqesitm": "interaction_text_easy",
    "seqesitm": "side_effect_text_easy",
    "depositmethodqesitm": "storage_text_easy",
    "opende": "open_date",
    "updatede": "update_date",
    "itemimage": "item_image_url",
    "bizrno": "bizrno",
}

def to_snake_case(name: str) -> str:
    name = str(name).strip()
    if name in COLUMN_RENAME_MAP:
        return COLUMN_RENAME_MAP[name]

    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"__+", "_", name)
    return name.lower()

def clean_none_like(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        if s in NONE_LIKE:
            return None
    return v

def normalize_date_value(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None

    s = str(v).strip()
    if s in NONE_LIKE:
        return None

    # 20190909.0 -> 20190909
    if re.fullmatch(r"\d{8}\.0", s):
        s = s[:-2]

    # YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        try:
            dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            return None if pd.isna(dt) else dt.strftime("%Y-%m-%d")
        except:
            return s

    # ISO / 일반 날짜
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return s
        return dt.strftime("%Y-%m-%d")
    except:
        return s

def try_numeric(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in NONE_LIKE:
        return None

    # .0으로 끝나는 거 정수로
    if re.fullmatch(r"-?\d+\.0", s):
        return int(float(s))

    # 일반 실수/정수
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        num = float(s)
        if num.is_integer():
            return int(num)
        return num

    # 그 외 숫자로 변환 가능한 형태 (예: .5 -> 0.5)
    try:
        num = float(s)
        if num.is_integer():
            return int(num)
        return num
    except:
        return None  # 숫자 컬럼인데 숫자가 아니면 None 처리

def fix_broken_item_image(value):
    """
    낱알식별정보의 item_image_url이 깨져서 뒤에 다른 JSON 문자열이 섞인 경우,
    첫 URL만 추출.
    """
    if value is None:
        return None

    s = str(value).strip()
    if s in NONE_LIKE:
        return None

    # URL 추출
    m = re.search(r"https?://[^\s\"\],]+", s)
    if m:
        return m.group(0)

    return s

def normalize_dataframe(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    # 1) 컬럼명 통일
    df.columns = [to_snake_case(c) for c in df.columns]

    # 2) 값 정리
    for col in df.columns:
        df[col] = df[col].map(clean_none_like)

    # 3) 날짜 정리
    for col in df.columns:
        if col in DATE_COL_HINTS or col.endswith("_date"):
            df[col] = df[col].map(normalize_date_value)

    # 4) 숫자 정리
    for col in df.columns:
        if col in NUMERIC_COLS_HINTS:
            df[col] = df[col].map(try_numeric)

    # 5) 낱알식별정보 이미지 깨짐 보정
    if "item_image_url" in df.columns:
        df["item_image_url"] = df["item_image_url"].map(fix_broken_item_image)

    # 6) 허가상세 빈 문서값 정리
    for doc_col in ["ee_doc_data", "ud_doc_data", "nb_doc_data"]:
        if doc_col in df.columns:
            df[doc_col] = df[doc_col].map(lambda x: None if x in [None, ""] else x)

    # 7) EDI / STD_CD 같은 콤마구분 문자열은 공백 정리만
    for code_col in ["edi_code", "std_cd"]:
        if code_col in df.columns:
            df[code_col] = df[code_col].map(
                lambda x: None if x is None else ",".join([p.strip() for p in str(x).split(",") if p.strip()])
            )

    # 8) warning_type 있으면 정수화
    if "warning_type" in df.columns:
        df["warning_type"] = df["warning_type"].map(try_numeric)

    return df

# =========================
# 3) 파일 읽기 / 저장
# =========================

def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)

    elif suffix == ".parquet":
        return pd.read_parquet(path)

    elif suffix == ".json":
        # 1) 일반 json list/object
        # 2) 한 줄에 한 객체 형태도 대응
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()

        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            elif isinstance(obj, dict):
                return pd.DataFrame([obj])
        except:
            pass

        # json lines fallback
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    elif suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    else:
        raise ValueError(f"지원하지 않는 파일 형식: {path}")

def save_file(df: pd.DataFrame, src_path: Path, out_root: Path):
    rel_path = src_path.relative_to(INPUT_DIR)
    out_path = out_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    elif suffix == ".parquet":
        df.to_parquet(out_path, index=False)

    elif suffix in [".json", ".jsonl"]:
        # jsonl로 저장하면 RAG 파이프라인에서 다루기 편함
        out_jsonl = out_path.with_suffix(".jsonl")
        df.to_json(out_jsonl, orient="records", force_ascii=False, lines=True)
    else:
        out_csv = out_path.with_suffix(".csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# =========================
# 4) 전체 폴더 일괄 처리
# =========================

def main():
    exts = {".csv", ".json", ".jsonl", ".parquet"}
    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print("처리할 파일이 없습니다.")
        return

    print(f"총 {len(files)}개 파일 처리 시작")

    success = 0
    failed = []

    for path in files:
        try:
            df = load_file(path)
            if df.empty:
                print(f"[SKIP] 비어있음: {path.name}")
                continue

            df = normalize_dataframe(df, path.name)
            save_file(df, path, OUTPUT_DIR)

            print(f"[OK] {path.name}  ({len(df)} rows)")
            success += 1

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            failed.append((str(path), str(e)))

    print("\n=== 완료 ===")
    print(f"성공: {success}")
    print(f"실패: {len(failed)}")

    if failed:
        print("\n실패 목록:")
        for fp, err in failed:
            print(f"- {fp}: {err}")

if __name__ == "__main__":
    main()
