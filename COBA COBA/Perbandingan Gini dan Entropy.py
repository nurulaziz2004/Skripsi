import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Buat folder untuk menyimpan hasil
os.makedirs("hasil_output", exist_ok=True)

# 1. Baca dataset
file_path = "D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\CAPSTONE PROJECT\Datasets\dataset_selada_no_age.csv"  # ganti sesuai lokasi file Anda
data = pd.read_csv(file_path)

FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET = "label"

X = data[FEATURES]
y = data[TARGET]

# 2. Split data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 3. Uji perbandingan Gini vs Entropy
criterions = ["gini", "entropy"]

for crit in criterions:
    print("\n==============================")
    print(f"Training Decision Tree dengan criterion = {crit}")
    print("==============================")

    # Buat model
    model = DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")
    report = classification_report(y_test, y_pred, target_names=["Tidak Siram", "Siram"])
    print("\nClassification Report:")
    print(report)

    # Simpan classification report ke file
    with open(f"hasil_output/classification_report_{crit}.txt", "w") as f:
        f.write(f"Criterion = {crit}\n")
        f.write(f"Akurasi = {acc:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak Siram", "Siram"], yticklabels=["Tidak Siram", "Siram"])
    plt.title(f"Confusion Matrix (criterion = {crit})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"hasil_output/confusion_matrix_{crit}.png")
    plt.close()

    # Visualisasi pohon
    plt.figure(figsize=(14,7))
    plot_tree(model, feature_names=FEATURES, class_names=["Tidak Siram", "Siram"],
              filled=True, rounded=True, fontsize=10)
    plt.title(f"Decision Tree dengan criterion = {crit}")
    plt.tight_layout()
    plt.savefig(f"hasil_output/tree_{crit}.png")
    plt.close()

    # Simpan aturan pohon ke file txt
    rules_text = export_text(model, feature_names=FEATURES)
    with open(f"hasil_output/tree_rules_{crit}.txt", "w") as f:
        f.write(rules_text)

    # Feature Importance
    importances = model.feature_importances_
    feat_importance = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_importance, y=feat_importance.index)
    plt.title(f"Feature Importance (criterion = {crit})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(f"hasil_output/feature_importance_{crit}.png")
    plt.close()

print("\nSemua hasil evaluasi tersimpan di folder 'hasil_output'.")
