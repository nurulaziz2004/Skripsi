# pip install paho-mqtt


import paho.mqtt.client as mqtt

# ====== Konfigurasi broker MQTT ======
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"

# ====== Topik sensor & relay ======
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

# ====== Callback ======
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    # Subscribe semua sensor + relay
    for t in sensor_topics:
        client.subscribe(t)
    client.subscribe(relay1)
    client.subscribe(relay2)
    client.subscribe(relay3)

def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")

# ====== Setup client ======
client = mqtt.Client(client_id="SatriaSensors_Python", protocol=mqtt.MQTTv311)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

# ====== Contoh kirim perintah relay ======
def kirim_relay(relay, state):
    if relay == 1:
        client.publish(relay1, state)
    elif relay == 2:
        client.publish(relay2, state)
    elif relay == 3:
        client.publish(relay3, state)
    print(f"Relay {relay} => {state}")

# ====== Jalankan loop ======
client.loop_start()

# Contoh tes: nyalakan/matikan relay
import time
time.sleep(2)

kirim_relay(1, "ON")
time.sleep(2)
kirim_relay(1, "OFF")

kirim_relay(2, "ON")
time.sleep(2)
kirim_relay(2, "OFF")

kirim_relay(3, "ON")
time.sleep(2)
kirim_relay(3, "OFF")

# tetap dengarkan pesan
while True:
    time.sleep(1)
