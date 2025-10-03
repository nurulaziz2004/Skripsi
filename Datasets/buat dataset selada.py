import os
import random
import pandas as pd

def generate_selada_no_age_csv(
    out_path="dataset_selada_no_age.csv",
    max_rows=5000,
    noise=0.5,
    seed=None
):
    """
    Generate a synthetic lettuce dataset (no 'umur' column).
    Save CSV with columns: suhu, kelembaban, kelembaban_tanah, intensitas_cahaya, label
    """
    if seed is not None:
        random.seed(seed)

    rows = []
    for _ in range(max_rows):
        # realistic ranges for lettuce
        suhu = random.uniform(15.0, 35.0)                # °C
        kelembaban = random.uniform(30.0, 90.0)          # % RH
        kelembaban_tanah = random.uniform(15.0, 80.0)    # % soil moisture
        intensitas_cahaya = random.uniform(10.0, 100.0)  # relative 0-100

        # Decision rule (no age)
        if (kelembaban_tanah < 40) \
           or (suhu > 28 and kelembaban < 55 and intensitas_cahaya > 70) \
           or (kelembaban < 45 and kelembaban_tanah < 50):
            label = 1
        else:
            label = 0

        # add small realistic noise
        suhu += random.uniform(-noise, noise)
        kelembaban += random.uniform(-noise, noise)
        kelembaban_tanah += random.uniform(-noise, noise)
        intensitas_cahaya += random.uniform(-noise, noise)

        rows.append({
            "suhu": round(suhu, 2),
            "kelembaban": round(kelembaban, 2),
            "kelembaban_tanah": round(kelembaban_tanah, 2),
            "intensitas_cahaya": round(intensitas_cahaya, 2),
            "label": label
        })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Dataset selesai dibuat: {out_path}")
    print("Jumlah baris:", len(df))
    print(df['label'].value_counts().rename_axis('label').reset_index(name='count'))
    return df

if __name__ == "__main__":
    # Ubah parameter sesuai kebutuhan (contoh: max_rows=10000)
    df = generate_selada_no_age_csv(out_path="dataset_selada_no_age.csv", max_rows=5000, noise=0.5, seed=42)
