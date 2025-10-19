import os
import time
import threading
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

try :
    import paho.mqtt.client as mqtt
    from flask import Flask, render_template_string, request, jsonify
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
except :
    print("ADA LIBRARY GAGAL, MULAI MENGINSTAL ..... ")
    time.sleep(5)
    os.system("pip install paho-mqtt==2.1.0 flask==3.1.1 scikit-learn==1.5.2 matplotlib==3.9.2 pandas==2.2.3")
    import paho.mqtt.client as mqtt
    from flask import Flask, render_template_string, request, jsonify
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # gunakan backend non-GUI




# ===================== Konfigurasi =====================
BROKER = "103.186.1.210"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"
current_folder = os.path.dirname(os.path.abspath(__file__))
print("current FOLDER : ",current_folder)


# Topik relay
RELAY_TOPICS = {
    1: f"{TOPIC_BASE}/relay1",
    2: f"{TOPIC_BASE}/relay2",
    3: f"{TOPIC_BASE}/relay3",
}

# Topik sensor
sensor_topics = [
    f"{TOPIC_BASE}/ldr",
    f"{TOPIC_BASE}/suhu",
    f"{TOPIC_BASE}/kelembaban",
    f"{TOPIC_BASE}/kelembaban_tanah_1",
    f"{TOPIC_BASE}/kelembaban_tanah_2",
    f"{TOPIC_BASE}/kelembaban_tanah_3",
]

SENSOR_KEYS = [t.split("/")[-1] for t in sensor_topics]

sensor_data = {k: None for k in SENSOR_KEYS}
sensor_timestamp = {k: None for k in SENSOR_KEYS}
control_mode = {"mode": "manual"}
relay_state = {1: "OFF", 2: "OFF", 3: "OFF"}
decision_info = {
    "ready": False,
    "features": {},
    "prediction": {},
    "proba": {},
    "path_rules": {},
    "updated_at": None,
}

# ===================== MQTT =====================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    if rc == 0:
        print("[MQTT] Connection successful! Subscribing to topics...")
        for t in sensor_topics:
            client.subscribe(t)
            print(f"  ✓ Subscribed: {t}")
        for t in RELAY_TOPICS.values():
            client.subscribe(t)
            print(f"  ✓ Subscribed: {t}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def safe_float(s):
    try:
        return float(str(s).strip())
    except:
        return None

def on_disconnect(client, userdata, rc):
    """Callback saat koneksi terputus"""
    if rc != 0:
        print(f"[MQTT] Unexpected disconnect! RC: {rc}. Auto-reconnecting...")
        # Paho MQTT akan auto-reconnect karena loop_start() aktif

def on_publish(client, userdata, mid):
    """Callback saat publish berhasil dikirim ke broker"""
    pass  # Matikan log untuk mengurangi overhead

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")
    key = topic.split("/")[-1]

    if key in sensor_data:
        val = safe_float(payload)
        sensor_data[key] = val
        sensor_timestamp[key] = datetime.now().strftime("%H:%M:%S")
    for rid, rtopic in RELAY_TOPICS.items():
        if topic == rtopic:
            print(f"[RELAY-ECHO] relay{rid} <- {payload} (ESP32 terima!)")

client = mqtt.Client(client_id="SatriaSensors_FlaskDT", protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_disconnect = on_disconnect  # Auto-reconnect handler
client.on_message = on_message
client.on_publish = on_publish  # Tambah callback untuk track publish

# Koneksi MQTT dan tunggu sampai ready
print(f"[MQTT] Connecting to {BROKER}:{PORT}...")
client.connect(BROKER, PORT, keepalive=30)  # Ubah dari 60 ke 30 detik - lebih responsif
client.loop_start()

# TUNGGU sampai koneksi berhasil (maksimal 10 detik)
mqtt_ready = False
for i in range(50):  # 50 x 0.2s = 10 detik
    if client.is_connected():
        mqtt_ready = True
        print(f"[MQTT] Connected and ready! ✓")
        break
    time.sleep(0.2)

if not mqtt_ready:
    print("[MQTT] WARNING: Connection timeout, but continuing...")

def publish_relay(relay_id: int, state: str):
    start_time = time.time()
    state = "ON" if str(state).upper() == "ON" else "OFF"
    topic = RELAY_TOPICS.get(relay_id)
    if topic:
        # METODE 1: Pakai paho-mqtt (client yang sudah ada)
        result = client.publish(topic, state, qos=0, retain=False)
        relay_state[relay_id] = state
        elapsed = (time.time() - start_time) * 1000
        print(f"[RELAY] relay{relay_id} => {state} | Publish time: {elapsed:.2f}ms | RC: {result.rc}")

# Thread pool untuk non-blocking publish
publish_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="MQTTPub")

def _do_publish(relay_id: int, state: str, topic: str):
    """Internal function untuk publish di background thread"""
    try:
        # Tunggu jika client belum connected
        max_wait = 20  # 2 detik max
        for i in range(max_wait):
            if client.is_connected():
                break
            time.sleep(0.1)
        
        if not client.is_connected():
            print(f"[RELAY-BG-SKIP] relay{relay_id} => {state} (not connected)")
            return
            
        info = client.publish(topic, state, qos=0)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[RELAY-BG] relay{relay_id} => {state} sent ✓")
        else:
            print(f"[RELAY-BG] relay{relay_id} => {state} RC: {info.rc}")
    except Exception as e:
        print(f"[RELAY-BG-ERROR] relay{relay_id}: {e}")

def publish_relay_direct(relay_id: int, state: str):
    """Publish NON-BLOCKING di background thread - return instant!"""
    start = time.time()
    state = "ON" if str(state).upper() == "ON" else "OFF"
    topic = RELAY_TOPICS.get(relay_id)
    relay_state[relay_id] = state  # Update state langsung
    
    if topic:
        # Submit ke thread pool - return INSTANT tanpa tunggu
        publish_executor.submit(_do_publish, relay_id, state, topic)
        elapsed = (time.time() - start) * 1000
        print(f"[RELAY-INSTANT] relay{relay_id} => {state} | Time: {elapsed:.2f}ms (queued) ✓")

# ===================== Decision Tree =====================

df = pd.read_csv(os.path.join(current_folder, "dataset_selada_no_age.csv"))
FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))

os.makedirs(os.path.join(current_folder,"static"), exist_ok=True)
plt.figure(figsize=(13, 7))
plot_tree(model, feature_names=FEATURES, class_names=["Tidak Siram","Siram"], filled=True, rounded=True)
plt.tight_layout()
plt.savefig(os.path.join(current_folder,"static/tree.png"), dpi=140)
plt.close()

tree_rules_text = export_text(model, feature_names=FEATURES)

# ===================== Decision Path =====================
def decision_path_rules(clf: DecisionTreeClassifier, xrow: np.ndarray, features: list):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold
    node_indicator = clf.decision_path(xrow.reshape(1, -1))
    node_index = node_indicator.indices
    rules = []
    for node_id in node_index:
        if feature[node_id] != -2:
            fname = features[feature[node_id]]
            thr = threshold[node_id]
            if xrow[feature[node_id]] <= thr:
                rules.append(f"{fname} <= {thr:.2f}")
            else:
                rules.append(f"{fname} > {thr:.2f}")
    return rules

# ===================== Loop Otomatis =====================
def collect_latest_features():
    suhu = sensor_data.get("suhu")
    kelembaban = sensor_data.get("kelembaban")
    ldr = sensor_data.get("ldr")
    pot_features = {}
    for i in range(1,4):
        tanah = sensor_data.get(f"kelembaban_tanah_{i}")
        pot_features[f"pot_{i}"] = {
            "suhu": suhu,
            "kelembaban": kelembaban,
            "kelembaban_tanah": tanah,
            "intensitas_cahaya": ldr
        }
    if any(any(v is None for v in pot_features[p].values()) for p in pot_features):
        return None
    return pot_features

def auto_control_loop():
    while True:
        try:
            if control_mode["mode"] == "auto":
                feats = collect_latest_features()
                if feats is not None:
                    for i in range(1,4):
                        x = np.array([feats[f"pot_{i}"][f] for f in FEATURES], dtype=float)
                        pred = int(model.predict([x])[0])
                        proba = model.predict_proba([x])[0].tolist() if hasattr(model, "predict_proba") else None
                        rules = decision_path_rules(model, x, FEATURES)

                        decision_info["features"][f"pot_{i}"] = feats[f"pot_{i}"]
                        decision_info["prediction"][f"pot_{i}"] = pred
                        decision_info["proba"][f"pot_{i}"] = proba
                        decision_info["path_rules"][f"pot_{i}"] = rules

                        # Kontrol relay masing-masing pot
                        publish_relay(i, "ON" if pred==1 else "OFF")

                    decision_info["ready"] = True
                    decision_info["updated_at"] = datetime.now().strftime("%H:%M:%S")
            time.sleep(0.5)
        except Exception as e:
            print("[AUTO LOOP ERROR]", e)
            time.sleep(1)

threading.Thread(target=auto_control_loop, daemon=True).start()

# ===================== Flask =====================
app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>🌱 Smart Farming Dashboard</title>
<style>
body { font-family: Arial; margin:18px; }
.row { display:flex; flex-wrap: wrap; gap:12px; }
.card { border:1px solid #ddd; border-radius:12px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.sensor { width:220px; }
.btn { padding:8px 12px; border:1px solid #333; border-radius:8px; background:#f7f7f7; cursor:pointer; margin-right:6px; }
.btn.active { background:#e8ffe8; border-color:#2b8a3e; }
.muted { color:#777; font-size:12px; }
pre { background:#f5f5f5; padding:10px; border-radius:8px; overflow:auto; }
.tag { display:inline-block; padding:2px 6px; border-radius:6px; background:#eef; margin-right:6px; font-size:12px; }
</style>
<script>
function setMode(mode){
  fetch('/mode/'+mode, {method:'POST'});
  refreshAll();
}

function relay(rid,state){
  // Update UI INSTANT - tidak tunggu apapun!
  document.getElementById('r'+rid).innerText=state;
  document.getElementById('r'+rid+'_on').classList.toggle('active', state=='ON');
  document.getElementById('r'+rid+'_off').classList.toggle('active', state=='OFF');
  
  // Kirim ke server pakai beacon API - tercepat di browser
  if (navigator.sendBeacon) {
    navigator.sendBeacon(`/relay/${rid}/${state}`);
  } else {
    // Fallback untuk browser lama
    new Image().src = `/relay/${rid}/${state}?_=${+new Date()}`;
  }
}

async function refreshSensors() {
  let r = await fetch('/sensors');
  let d = await r.json();
  let html = "";
  Object.keys(d.values).forEach(k => {
    let val = d.values[k] ?? '-';
    let unit = (k.toLowerCase().includes("suhu") || k.toLowerCase().includes("temp")) ? "°C" : "%";  
    html += `<div class="card sensor"><b>${k}</b><br>${val}${unit}<br></div>`;
  });

  document.getElementById('sensors').innerHTML = html;
  document.getElementById('mode').innerText = d.mode.toUpperCase();
  document.getElementById('btn_manual').classList.toggle('active', d.mode=='manual');
  document.getElementById('btn_auto').classList.toggle('active', d.mode=='auto');
  ['1','2','3'].forEach(rid=>{
    let state=d.relay[rid];
    document.getElementById('r'+rid).innerText=state;
    document.getElementById('r'+rid+'_on').classList.toggle('active', state=='ON');
    document.getElementById('r'+rid+'_off').classList.toggle('active', state=='OFF');
  });
}
async function refreshDecision(){
  let r = await fetch('/decision'); let d = await r.json();
  let info = document.getElementById('decision');
  if(!d.ready){ info.innerHTML='<i>Menunggu data lengkap...</i>'; return; }
  let html="";
  for(let i=1;i<=3;i++){
    let p='pot_'+i;
    let feats = d.features[p];
    let proba = (d.proba[p])? `P(0)=${d.proba[p][0].toFixed(3)}, P(1)=${d.proba[p][1].toFixed(3)}`:'';
    let rules = (d.path_rules[p]||[]).map(r=>"• "+r).join('<br>');
    html+=`<h4>Pot ${i}</h4>
      <div><b>Prediksi:</b> ${d.prediction[p]==1?'SIRAM':'TIDAK SIRAM'}</div>
      <div>${proba}</div>
      <div style="margin-top:6px">${Object.entries(feats).map(([k,v])=>`<span class="tag">${k}:${v.toFixed(2)}</span>`).join(' ')}</div>
      <div class="muted" style="margin-top:4px">Updated: ${d.updated_at||''}</div>
      <hr>
      <div><b>Jalur Keputusan:</b><br>${rules}</div>`;
  }
  info.innerHTML=html;
}
async function refreshRules(){
  let r = await fetch('/rules'); let t = await r.text();
  document.getElementById('rules').innerText=t;
  document.getElementById('tree').src='/static/tree.png?ts='+Date.now();
}
async function refreshAll(){ 
  refreshSensors(); 
  refreshDecision(); 
  // refreshRules(); // Jangan refresh rules terus-menerus, hanya sekali saat load
}
// Refresh cepat untuk sensor & decision, rules hanya sekali
setInterval(refreshAll,1000); // Ubah dari 500ms ke 1000ms untuk kurangi beban
window.onload=function(){ 
  refreshAll(); 
  refreshRules(); // Rules hanya load sekali saat pertama
};
</script>
</head>
<body>
<h1>🌱 Smart Farming Dashboard</h1>

<div class="card" style="margin-bottom:12px;">
Mode: <b id="mode">-</b>
<button class="btn" id="btn_manual" onclick="setMode('manual')">Manual</button>
<button class="btn" id="btn_auto" onclick="setMode('auto')">Otomatis</button>
<span class="muted">Manual: pakai tombol relay. Otomatis: decision tree.</span>
</div>

<h2>Sensor</h2>
<div id="sensors" class="row"></div>

<h2>Kontrol Relay</h2>
<div class="card">
<div>Relay 1: <b id="r1">-</b>
<button class="btn" id="r1_on" onclick="relay(1,'ON')">ON</button>
<button class="btn" id="r1_off" onclick="relay(1,'OFF')">OFF</button></div>
<div>Relay 2: <b id="r2">-</b>
<button class="btn" id="r2_on" onclick="relay(2,'ON')">ON</button>
<button class="btn" id="r2_off" onclick="relay(2,'OFF')">OFF</button></div>
<div>Relay 3: <b id="r3">-</b>
<button class="btn" id="r3_on" onclick="relay(3,'ON')">ON</button>
<button class="btn" id="r3_off" onclick="relay(3,'OFF')">OFF</button></div>
<div class="muted" style="margin-top:6px">Catatan: Mode Otomatis mengatur relay sesuai prediksi.</div>
</div>

<h2>Decision Insight (Realtime)</h2>
<div id="decision" class="card"></div>

<h2>Decision Tree</h2>
<img id="tree" src="/static/tree.png" width="900" style="border:1px solid #ddd; border-radius:8px;">
<h3>Aturan Pohon</h3>
<pre id="rules"></pre>
</body>
</html>
"""

@app.route("/")
def index(): return render_template_string(PAGE)

@app.route("/mode/<mode>", methods=["POST", "GET"])
def set_mode(mode):
    control_mode["mode"] = "auto" if mode.lower().strip()=="auto" else "manual"
    return "", 204  # No Content - lebih cepat dari jsonify

@app.route("/relay/<int:rid>/<state>", methods=["POST", "GET"])
def set_relay(rid,state):
    # PAKAI MOSQUITTO_PUB LANGSUNG - seperti command terminal Anda!
    publish_relay_direct(rid, state)
    return "", 204  # No Content - response tercepat

@app.route("/sensors")
def sensors_api():
    return jsonify({"values": sensor_data, "timestamps": sensor_timestamp,
                    "mode": control_mode["mode"], "relay": relay_state})

@app.route("/decision")
def decision_api(): return jsonify(decision_info)

@app.route("/rules")
def rules_api(): return tree_rules_text

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
