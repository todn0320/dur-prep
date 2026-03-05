import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re

# =========================
# 1. 환경 설정
# =========================
# 작업 환경에 맞춰 BASE_DIR를 조정하세요. (현재는 상위 3단계 위 기준)
BASE_DIR = Path(__file__).resolve().parents[3]
IN_DIR = BASE_DIR / "processed/pc4/clean_by_type"
OUT_DIR = BASE_DIR / "processed/pc4/clean_by_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 정제 및 섹션 추출 유틸리티
# =========================

def clean_xml_to_text(xml_content):
    """XML 태그를 제거하고 내부의 깨끗한 텍스트만 반환"""
    if pd.isna(xml_content) or xml_content == "":
        return ""
    
    # 1. CDATA 내부 텍스트 추출
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", str(xml_content), flags=re.DOTALL)
    
    # 2. XML 태그들 제거 (제목 속성 등은 사라지고 내용만 남음)
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. 연속된 공백, 줄바꿈, 탭 등을 하나의 공백으로 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def extract_section(xml_content, section_title_keywords):
    """원본 XML에서 특정 섹션(경고, 금기 등)만 따로 추출"""
    if pd.isna(xml_content) or xml_content == "":
        return None
    
    # ARTICLE title 속성을 기준으로 분할 추출
    articles = re.findall(r'<ARTICLE title="(.*?)">(.*?)</ARTICLE>', str(xml_content), flags=re.DOTALL)
    
    results = []
    for title, content in articles:
        if any(kw in title for kw in section_title_keywords):
            results.append(clean_xml_to_text(content))
    
    return "\n".join(results) if results else None

# =========================
# 3. 메인 실행부
# =========================

def main():
    # 파일명 확인: 원본 clean 파일을 읽어서 cleantext 파일로 저장
    target_file = IN_DIR / "drugPrmsnInfo_의약품제품허가정보_clean.parquet"
    out_path = OUT_DIR / "drugPrmsnInfo_의약품제품허가정보_cleantext.parquet"

    if not target_file.exists():
        print(f"❌ 입력 파일 없음: {target_file}")
        return

    try:
        print(f"▶ [허가정보] Clean Text 변환 및 섹션 분리 시작...")
        df = pd.read_parquet(target_file)
        
        # 1. 기본 컬럼 정제
        df["ITEM_SEQ"] = df["ITEM_SEQ"].astype(str).str.strip()
        df["ETC_OTC_CODE"] = df["ETC_OTC_CODE"].fillna("").str.strip()
        
        # 2. NB_DOC_DATA에서 핵심 섹션 선제적 추출 (원본이 Clean되기 전에 수행해야 함)
        print("   - 주요 NB 섹션(경고/금기/신중/이상반응) 추출 중...")
        df["warning_text"] = df["NB_DOC_DATA"].apply(lambda x: extract_section(x, ["경고"]))
        df["contraindication_text"] = df["NB_DOC_DATA"].apply(lambda x: extract_section(x, ["투여하지 말 것"]))
        df["caution_text"] = df["NB_DOC_DATA"].apply(lambda x: extract_section(x, ["신중히 투여"]))
        df["adverse_text"] = df["NB_DOC_DATA"].apply(lambda x: extract_section(x, ["이상반응"]))

        # 3. [핵심] 원본 XML 컬럼들을 Clean Text로 덮어쓰기
        print("   - 원본 XML 컬럼들을 Clean Text로 변환 중...")
        doc_cols = ["EE_DOC_DATA", "UD_DOC_DATA", "NB_DOC_DATA"]
        for col in doc_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_xml_to_text)

        # 4. 최종 저장
        df.to_parquet(
            out_path,
            engine='pyarrow',
            index=False,
            compression='zstd'
        )

        print(f"✅ 정제 및 변환 완료: {out_path.name}")
        print(f"✅ 결과 컬럼: {doc_cols} (Cleaned) + warning, contraindication, etc.")
        
        # 데이터 검증 샘플 출력
        sample = df.iloc[0]
        print("\n--- [Clean Text 변환 결과 샘플] ---")
        print(f"상품명: {sample['ITEM_NAME']}")
        print(f"효능효과(EE)일부: {sample['EE_DOC_DATA'][:100]}...")
        print(f"이상반응 섹션일부: {str(sample['adverse_text'])[:100]}...")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    main()