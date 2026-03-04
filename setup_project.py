import os

for d in [
    'data/raw',
    'data/interim',
    'data/processed',
    'data/processed/pc2',
    'data/processed/pc3',
    'data/processed/pc4',
    'data/processed/pc5',
    'src/schema',
    'docs',
    'logs',
    'release/day1'
]:
    os.makedirs(d, exist_ok=True)

print("폴더 구조 완료")