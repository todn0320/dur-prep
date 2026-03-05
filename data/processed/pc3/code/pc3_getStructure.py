
import json
json_path = 'C:/Users/soldesk/Desktop/dur-prep/data/raw/eDrugPrmsnInfo/낱알식별정보.json' # ← 실제 경로로 수정 !
with open(json_path, encoding='utf-8') as f:
 raw = json.load(f)
# 최상위 구조 확인
if isinstance(raw, list):
 print(' 📋 형태: 리스트')
 print('총 항목 수 :', len(raw))
 print('첫 항목 키 :', list(raw[0].keys()))
elif isinstance(raw, dict):
    print(' 📋 형태: dict')
    for k, v in raw.items():
        print(f' 키: {k} → {type(v).__name__}', end='')
        if isinstance(v, list): print(f' ({len(v)}개)')
        else: print()
