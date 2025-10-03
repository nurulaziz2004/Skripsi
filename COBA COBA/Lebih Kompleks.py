import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# Buat folder untuk menyimpan hasil
os.makedirs("hasil_output", exist_ok=True)

# 1. Baca dataset
file_path = r"D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\dataset_selada_no_age.csv"
data = pd.read_csv(file_path)

FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET = "label"

X = data[FEATURES]
y = data[TARGET]

# 2. Exploratory Data Analysis (EDA)
# Distribusi label
plt.figure(figsize=(5,4))
sns.countplot(x=y, palette="Set2")
plt.title("Distribusi Label (0=Tidak Siram, 1=Siram)")
plt.xlabel("Label")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig("hasil_output/distribusi_label.png")
plt.close()

# Distribusi setiap fitur
for col in FEATURES:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True, bins=20, color="teal")
    plt.title(f"Distribusi {col}")
    plt.tight_layout()
    plt.savefig(f"hasil_output/distribusi_{col}.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=y, y=data[col], palette="Set3")
    plt.title(f"Boxplot {col} terhadap Label")
    plt.xlabel("Label")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"hasil_output/boxplot_{col}.png")
    plt.close()

# Heatmap korelasi
plt.figure(figsize=(7,6))
sns.heatmap(data[FEATURES+[TARGET]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Antar Variabel")
plt.tight_layout()
plt.savefig("hasil_output/heatmap_korelasi.png")
plt.close()

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

criterions = ["gini", "entropy"]
akurasi_list = []

for crit in criterions:
    print("\n==============================")
    print(f"Training Decision Tree dengan criterion = {crit}")
    print("==============================")

    # Buat model
    model = DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    akurasi_list.append((crit, acc))
    print(f"Akurasi: {acc:.4f}")
    report = classification_report(y_test, y_pred, target_names=["Tidak Siram", "Siram"])
    print("\nClassification Report:")
    print(report)

    with open(f"hasil_output/classification_report_{crit}.txt", "w") as f:
        f.write(f"Criterion = {crit}\n")
        f.write(f"Akurasi = {acc:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Tidak Siram", "Siram"],
                yticklabels=["Tidak Siram", "Siram"])
    plt.title(f"Confusion Matrix (criterion = {crit})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"hasil_output/confusion_matrix_{crit}.png")
    plt.close()

    # Visualisasi Pohon
    plt.figure(figsize=(14,7))
    plot_tree(model, feature_names=FEATURES, class_names=["Tidak Siram", "Siram"],
              filled=True, rounded=True, fontsize=10)
    plt.title(f"Decision Tree dengan criterion = {crit}")
    plt.tight_layout()
    plt.savefig(f"hasil_output/tree_{crit}.png")
    plt.close()

    # Aturan pohon
    rules_text = export_text(model, feature_names=FEATURES)
    with open(f"hasil_output/tree_rules_{crit}.txt", "w") as f:
        f.write(rules_text)

    # Feature Importance
    importances = model.feature_importances_
    feat_importance = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_importance, y=feat_importance.index, palette="viridis")
    plt.title(f"Feature Importance (criterion = {crit})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(f"hasil_output/feature_importance_{crit}.png")
    plt.close()

    plt.figure(figsize=(6,6))
    plt.pie(feat_importance, labels=feat_importance.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("pastel"))
    plt.title(f"Feature Importance Pie (criterion = {crit})")
    plt.tight_layout()
    plt.savefig(f"hasil_output/feature_importance_pie_{crit}.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (criterion = {crit})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"hasil_output/roc_curve_{crit}.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, lw=2, color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (criterion = {crit})")
    plt.tight_layout()
    plt.savefig(f"hasil_output/pr_curve_{crit}.png")
    plt.close()

# Perbandingan akurasi antar criterion
df_acc = pd.DataFrame(akurasi_list, columns=["Criterion", "Accuracy"])
plt.figure(figsize=(6,4))
sns.barplot(x="Criterion", y="Accuracy", data=df_acc, palette="Set1")
plt.ylim(0,1)
plt.title("Perbandingan Akurasi Gini vs Entropy")
plt.tight_layout()
plt.savefig("hasil_output/perbandingan_akurasi.png")
plt.close()

# Learning Curve
plt.figure(figsize=(7,5))
train_sizes, train_scores, test_scores = learning_curve(
    DecisionTreeClassifier(max_depth=4, random_state=42),
    X, y, cv=5, train_sizes=np.linspace(0.1,1.0,10), n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
plt.plot(train_sizes, test_mean, "o-", color="red", label="Cross-validation score")
plt.xlabel("Jumlah Data Latih")
plt.ylabel("Skor Akurasi")
plt.title("Learning Curve Decision Tree")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("hasil_output/learning_curve.png")
plt.close()

print("\nSemua hasil evaluasi + grafik tambahan tersimpan di folder 'hasil_output'.")
