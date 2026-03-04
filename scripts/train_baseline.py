import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "output" / "final" / "train_dataset.csv"

print("데이터 로딩...")
df = pd.read_csv(DATA_PATH)

print("총 샘플 수:", len(df))

X = df["text"]
y = df["label"]

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("TF-IDF 벡터화...")
vectorizer = TfidfVectorizer(max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("모델 학습...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

print("평가...")
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))