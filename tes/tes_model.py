from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Misal dataset sudah ada di CSV
df = pd.read_csv("D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\CAPSTONE PROJECT\Datasets\dataset_selada_no_age.csv")

X = df[["suhu","kelembaban","kelembaban_tanah","intensitas_cahaya"]]  # fitur
y = df["label"]  # target

# Bagi data jadi train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Latih decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))


# tampilkan struktur pohon
tree_rules = export_text(model, feature_names=list(X_train.columns))
print(tree_rules)