import os
import random
import time


current_folder = os.getcwd()
print("Folder sekarang : ", current_folder)
path_dataset = os.path.join(current_folder,"/dataset.csv")

jumlah_on = 0
jumlah_off = 0
max_data = 50000  # batas jumlah baris dataset
noise = 0.1



label = 0
last_label = 0

# Buat header kalau file belum ada
if not os.path.exists(path_dataset):
    with open(path_dataset, "w") as f:
        f.write("suhu,kelembaban,kelembaban_tanah,intensitas_cahaya,label\n")

for _ in range(max_data):
    suhu = random.uniform(0.0, 100.0)
    kelembaban = random.uniform(0.0, 100.0)
    kelembaban_tanah = random.uniform(0.0, 100.0)
    intensitas_cahaya = random.uniform(0.0, 100.0)

    # Aturan penyiraman sederhana
    if (kelembaban_tanah < 30 or (suhu > 30 and kelembaban < 50 and intensitas_cahaya > 70)):
        label = 1  # Siram
        suhu += random.uniform(-noise, noise)
        kelembaban += random.uniform(-noise, noise)
        kelembaban_tanah += random.uniform(-noise, noise)
        intensitas_cahaya += random.uniform(-noise, noise)
        jumlah_on += 1
    else:
        label = 0  # Tidak siram
        suhu += random.uniform(-noise, noise)
        kelembaban += random.uniform(-noise, noise)
        kelembaban_tanah += random.uniform(-noise, noise)
        intensitas_cahaya += random.uniform(-noise, noise)
        jumlah_off += 1

    if jumlah_off > jumlah_on :
        jumlah_off = jumlah_on

    if label != last_label :
        data_baris = f"{suhu:.2f},{kelembaban:.2f},{kelembaban_tanah:.2f},{intensitas_cahaya:.2f},{label}\n"
        with open(path_dataset, "a") as f:
                f.write(data_baris)
        print(data_baris.strip(),"/" ,jumlah_on,"/" ,jumlah_off)

    # time.sleep(0.1)  # jeda kecil supaya lebih aman

    last_label = label
