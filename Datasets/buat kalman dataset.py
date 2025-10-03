import os
import random
import pandas as pd

class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, initial_estimate=50.0):
        self.process_variance = process_variance      # Q
        self.measurement_variance = measurement_variance  # R
        self.estimate = initial_estimate              # x_hat
        self.error_covariance = 1.0                   # P

    def update(self, measurement):
        # Prediction update
        self.error_covariance += self.process_variance

        # Kalman gain
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)

        # Measurement update
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error_covariance *= (1 - kalman_gain)

        return self.estimate


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

    # buat instance Kalman Filter khusus kelembaban_tanah
    kalman_soil = SimpleKalmanFilter(initial_estimate=40.0)

    rows = []
    for _ in range(max_rows):
        # realistic ranges for lettuce
        suhu = random.uniform(15.0, 35.0)                # °C
        kelembaban = random.uniform(30.0, 90.0)          # % RH
        kelembaban_tanah = random.uniform(15.0, 80.0)    # % soil moisture
        intensitas_cahaya = random.uniform(10.0, 100.0)  # relative 0-100

        # add small realistic noise
        suhu += random.uniform(-noise, noise)
        kelembaban += random.uniform(-noise, noise)
        kelembaban_tanah += random.uniform(-noise, noise)
        intensitas_cahaya += random.uniform(-noise, noise)

        # apply Kalman filter ke sensor soil moisture
        kelembaban_tanah_filtered = kalman_soil.update(kelembaban_tanah)

        # Decision rule (pakai nilai hasil filter)
        if (kelembaban_tanah_filtered < 40) \
           or (suhu > 28 and kelembaban < 55 and intensitas_cahaya > 70) \
           or (kelembaban < 45 and kelembaban_tanah_filtered < 50):
            label = 1
        else:
            label = 0

        rows.append({
            "suhu": round(suhu, 2),
            "kelembaban": round(kelembaban, 2),
            "kelembaban_tanah": round(kelembaban_tanah_filtered, 2),  # pakai hasil filter
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
    df = generate_selada_no_age_csv(
        out_path="dataset_selada_no_age_filtered.csv",
        max_rows=5000,
        noise=0.5,
        seed=42
    )
