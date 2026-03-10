# DUR 기반 의약품 안전정보 AI

## 프로젝트 목적
공공데이터(DUR, 허가정보, OTC)를 활용하여
약물 금기/주의 정보를 자동 분석하는 AI 시스템

## 데이터
- DUR 품목정보
- DUR 성분정보
- 의약품 허가정보
- e약은요

## 구조
data/
scripts/
src/

## 실행 방법
pip install -r requirements.txt
python scripts/train_baseline.py


api = 민욱님
db = 종성님
inference = 다현님
ocr = 태영님
pipeline = 형우님