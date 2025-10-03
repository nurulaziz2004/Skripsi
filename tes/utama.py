# ====== Install dulu ======
# pip install paho-mqtt flask scikit-learn matplotlib pandas

import os
import time
import random
import paho.mqtt.client as mqtt
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ====== Konfigurasi broker MQTT ======
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"

# ====== Topik relay ======
relay1 = f"{TOPIC_BASE}/relay1"
relay2 = f"{TOPIC_BASE}/relay2"
relay3 = f"{TOPIC_BASE}/relay3"

sensor_topics = [
    f"{TOPIC_BASE}/ldr",
    f"{TOPIC_BASE}/suhu",
    f"{TOPIC_BASE}/kelembaban",
    f"{TOPIC_BASE}/kelembaban_tanah_1",
    f"{TOPIC_BASE}/kelembaban_tanah_2",
    f"{TOPIC_BASE}/kelembaban_tanah_3"
]

# ====== Data sensor global ======
sensor_data = {t.split("/")[1]: None for t in sensor_topics}

# ====== Callback MQTT ======
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    for t in sensor_topics:
        client.subscribe(t)
    client.subscribe(relay1)
    client.subscribe(relay2)
    client.subscribe(relay3)

def on_message(client, userdata, msg):
    value = msg.payload.decode()
    sensor_data[msg.topic.split("/")[1]] = value
    print(f"[{msg.topic}] {value}")

# ====== Setup MQTT ======
client = mqtt.Client(client_id="SatriaSensors_Python", protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_start()

# ====== Kirim relay ======
def kirim_relay(relay, state):
    topic = {1: relay1, 2: relay2, 3: relay3}.get(relay, None)
    if topic:
        client.publish(topic, state)
        print(f"Relay {relay} => {state}")

# ====== Training Decision Tree ======
df = pd.read_csv("D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\CAPSTONE PROJECT\Datasets\dataset_selada_no_age.csv")
X = df[["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Simpan visualisasi pohon
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=list(X_train.columns), class_names=["Tidak Siram", "Siram"], filled=True)
os.makedirs("static", exist_ok=True)
plt.savefig("static/tree.png")

tree_rules = export_text(model, feature_names=list(X_train.columns))
print(tree_rules)

# ====== Flask App ======
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Farming Dashboard</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .sensor-box { padding:10px; margin:5px; border:1px solid #ccc; display:inline-block; width:250px; }
        .relay-btn { padding:10px; margin:5px; }
    </style>
    <script>
        async function toggleRelay(relay, state) {
            await fetch(`/relay/${relay}/${state}`);
            alert(`Relay ${relay} => ${state}`);
        }
        async function refreshSensors() {
            let res = await fetch('/sensors');
            let data = await res.json();
            let container = document.getElementById('sensor-container');
            container.innerHTML = "";
            for (let key in data) {
                container.innerHTML += `<div class='sensor-box'><b>${key}</b>: ${data[key]}</div>`;
            }
        }
        setInterval(refreshSensors, 2000);
    </script>
</head>
<body>
    <h1>🌱 Smart Farming Dashboard</h1>

    <h2>Sensor Data</h2>
    <div id="sensor-container"></div>

    <h2>Relay Control</h2>
    <button class="relay-btn" onclick="toggleRelay(1,'ON')">Relay 1 ON</button>
    <button class="relay-btn" onclick="toggleRelay(1,'OFF')">Relay 1 OFF</button>
    <button class="relay-btn" onclick="toggleRelay(2,'ON')">Relay 2 ON</button>
    <button class="relay-btn" onclick="toggleRelay(2,'OFF')">Relay 2 OFF</button>
    <button class="relay-btn" onclick="toggleRelay(3,'ON')">Relay 3 ON</button>
    <button class="relay-btn" onclick="toggleRelay(3,'OFF')">Relay 3 OFF</button>

    <h2>Decision Tree</h2>
    <img src="/static/tree.png" width="800">
    <pre>{{ rules }}</pre>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE, rules=tree_rules)

@app.route("/sensors")
def sensors():
    return jsonify(sensor_data)

@app.route("/relay/<int:relay>/<state>")
def relay(relay, state):
    kirim_relay(relay, state)
    return f"Relay {relay} => {state}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
