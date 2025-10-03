# ====== Install ======
# pip install paho-mqtt flask scikit-learn pandas numpy graphviz
import re
import os, time, threading
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify
import paho.mqtt.client as mqtt
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# ===================== MQTT CONFIG =====================
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"

RELAY_TOPICS = {1: f"{TOPIC_BASE}/relay1",
                2: f"{TOPIC_BASE}/relay2",
                3: f"{TOPIC_BASE}/relay3"}

sensor_topics = [
    f"{TOPIC_BASE}/ldr",
    f"{TOPIC_BASE}/suhu",
    f"{TOPIC_BASE}/kelembaban",
    f"{TOPIC_BASE}/kelembaban_tanah_1",
    f"{TOPIC_BASE}/kelembaban_tanah_2",
    f"{TOPIC_BASE}/kelembaban_tanah_3"
]

SENSOR_KEYS = [t.split("/")[-1] for t in sensor_topics]

sensor_data = {k: None for k in SENSOR_KEYS}
sensor_timestamp = {k: None for k in SENSOR_KEYS}

relay_state = {1: "OFF", 2: "OFF", 3: "OFF"}
control_mode = {"mode": "manual"}

decision_info = {"ready": False,
                 "features": {},
                 "prediction": None,
                 "proba": None,
                 "path_rules": [],
                 "updated_at": None}

# ===================== MQTT CALLBACK =====================
def on_connect(client, userdata, flags, rc):
    print("Connected with code", rc)
    for t in sensor_topics:
        client.subscribe(t)
    for t in RELAY_TOPICS.values():
        client.subscribe(t)

def safe_float(s):
    try: return float(str(s).strip())
    except: return None

def on_message(client, userdata, msg):
    key = msg.topic.split("/")[-1]
    val = safe_float(msg.payload.decode())
    if key in sensor_data:
        sensor_data[key] = val
        sensor_timestamp[key] = datetime.now().strftime("%H:%M:%S")
    for rid, topic in RELAY_TOPICS.items():
        if msg.topic == topic:
            relay_state[rid] = str(msg.payload.decode())

client = mqtt.Client(client_id="SatriaSensors_FlaskGraphviz", protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_start()

def publish_relay(rid, state):
    state = "ON" if str(state).upper()=="ON" else "OFF"
    topic = RELAY_TOPICS.get(rid)
    if topic:
        client.publish(topic, state)
        relay_state[rid] = state

# ===================== DECISION TREE =====================
df = pd.read_csv("D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\CAPSTONE PROJECT\Datasets\dataset_selada_no_age.csv")
FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

tree_rules_text = "\n".join(export_graphviz(model, feature_names=FEATURES, class_names=["Tidak Siram","Siram"]).splitlines())

# ===================== GRAPHVIZ =====================
from graphviz import Source

def decision_path_rules(clf, xrow, features):
    tree = clf.tree_
    node_indicator = clf.decision_path([xrow])
    node_index = node_indicator.indices
    rules = []
    for node_id in node_index:
        f = tree.feature[node_id]
        thr = tree.threshold[node_id]
        if f != -2:
            fname = features[f]
            if xrow[f] <= thr: rules.append(f"{fname} <= {thr:.2f}")
            else: rules.append(f"{fname} > {thr:.2f}")
    return node_index, rules

def plot_tree_graphviz(clf, x_input, features, class_names, filename="static/tree", highlight_nodes=None, color="yellow"):
    from graphviz import Source
    os.makedirs("static", exist_ok=True)
    dot_data = export_graphviz(clf, feature_names=features, class_names=class_names,
                               filled=True, rounded=True, special_characters=True)
    highlight_nodes = highlight_nodes or set()
    def repl(match):
        node_id = int(match.group(1))
        rest = match.group(2)
        if node_id in highlight_nodes:
            return f'{node_id} [style=filled, fillcolor={color}{rest}]'
        return match.group(0)
    dot_data_active = re.sub(r'(\d+) (\[.*?\])', repl, dot_data)
    graph = Source(dot_data_active)
    graph.render(filename, format="png", cleanup=True)


# ===================== AUTO CONTROL =====================
def collect_latest_features():
    suhu = sensor_data.get("suhu")
    kelembaban = sensor_data.get("kelembaban")
    tanah_vals = [sensor_data.get("kelembaban_tanah_1"),
                  sensor_data.get("kelembaban_tanah_2"),
                  sensor_data.get("kelembaban_tanah_3")]
    tanah_valid = [v for v in tanah_vals if v is not None]
    if tanah_valid: kelembaban_tanah = float(np.mean(tanah_valid))
    else: kelembaban_tanah = None
    intensitas_cahaya = sensor_data.get("ldr")
    feats = {"suhu": suhu, "kelembaban": kelembaban,
             "kelembaban_tanah": kelembaban_tanah,
             "intensitas_cahaya": intensitas_cahaya}
    if any(v is None for v in feats.values()): return None
    return feats

def auto_loop():
    while True:
        try:
            if control_mode["mode"]=="auto":
                feats = collect_latest_features()
                if feats:
                    # Pastikan DataFrame 1 baris
                    x_df = pd.DataFrame([feats], columns=FEATURES)
                    pred = int(model.predict(x_df)[0])
                    proba = model.predict_proba(x_df)[0].tolist()
                    
                    # Ambil jalur node aktif
                    # Ambil node aktif
                    node_indicator = model.decision_path(x_df)
                    nodes_active = set(node_indicator.indices)

                    # Render tree
                    plot_tree_graphviz(model, x_df.iloc[0], FEATURES, ["Tidak Siram","Siram"],
                                    filename="static/tree", highlight_nodes=nodes_active, color="yellow")
                    rules = []
                    tree = model.tree_
                    for node_id in nodes_active:
                        f = tree.feature[node_id]
                        thr = tree.threshold[node_id]
                        if f != -2:
                            fname = FEATURES[f]
                            if x_df.iloc[0,f] <= thr:
                                rules.append(f"{fname} <= {thr:.2f}")
                            else:
                                rules.append(f"{fname} > {thr:.2f}")

                    decision_info.update({
                        "ready": True,
                        "features": feats,
                        "prediction": pred,
                        "proba": proba,
                        "path_rules": rules,
                        "updated_at": datetime.now().strftime("%H:%M:%S")
                    })

                    # Highlight node aktif dengan ungu
                    os.makedirs("static", exist_ok=True)
                    dot_data = export_graphviz(model, feature_names=FEATURES,
                                               class_names=["Tidak Siram","Siram"],
                                               filled=True, rounded=True,
                                               special_characters=True)
                    lines = dot_data.splitlines()
                    new_lines = []
                    node_id = 0
                    for line in lines:
                        if "shape=box" in line:
                            if node_id in nodes_active:
                               line = line.replace("style=filled", "style=filled, fillcolor=yellow")


                            node_id += 1
                        new_lines.append(line)
                    dot_data_active = "\n".join(new_lines)
                    from graphviz import Source
                    graph = Source(dot_data_active)
                    graph.render("static/tree", format="png", cleanup=True)

                    # Update relay otomatis
                    publish_relay(1, "ON" if pred==1 else "OFF")
            time.sleep(1)
        except Exception as e:
            print("[AUTO LOOP ERROR]", e)
            time.sleep(1)


threading.Thread(target=auto_loop, daemon=True).start()

# ===================== FLASK =====================
app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>🌱 Smart Farming Dashboard</title>
<style>
body{font-family:Arial;margin:18px}
.row{display:flex;flex-wrap:wrap;gap:12px}
.card{border:1px solid #ddd;border-radius:12px;padding:12px 14px;box-shadow:0 2px 6px rgba(0,0,0,0.05)}
.sensor{width:220px}.btn{padding:8px 12px;border:1px solid #333;border-radius:8px;background:#f7f7f7;cursor:pointer;margin-right:6px}
.muted{color:#777;font-size:12px}pre{background:#f5f5f5;padding:10px;border-radius:8px;overflow:auto}.tag{display:inline-block;padding:2px 6px;border-radius:6px;background:#eef;margin-right:6px;font-size:12px}
</style>
<script>
async function setMode(m){await fetch('/mode/'+m,{method:'POST'}); refreshAll();}
async function relay(r,s){await fetch(`/relay/${r}/${s}`,{method:'POST'}); refreshAll();}
async function refreshSensors(){let r=await fetch('/sensors'); let d=await r.json(); let html=""; Object.keys(d.values).forEach(k=>{html+=`<div class='card sensor'><b>${k}</b>: ${d.values[k]??'-'}<br><span class='muted'>${d.timestamps[k]??''}</span></div>`}); document.getElementById('sensors').innerHTML=html; document.getElementById('mode').innerText=d.mode.toUpperCase(); document.getElementById('r1').innerText=d.relay[1]; document.getElementById('r2').innerText=d.relay[2]; document.getElementById('r3').innerText=d.relay[3];}
async function refreshDecision(){let r=await fetch('/decision'); let d=await r.json(); document.getElementById('decision').innerHTML=`<p>Updated: ${d.updated_at??'-'}</p>
<b>Prediction:</b> ${d.prediction??'-'} <br>
<b>Probabilities:</b> ${d.proba??'-'} <br>
<b>Path Rules:</b> ${d.path_rules?.map(x=>'<span class="tag">'+x+'</span>').join(' ')??''}
<br><img src='/static/tree.png?ts=${Date.now()}' width=800>`;}
function refreshAll(){refreshSensors();refreshDecision();}
setInterval(refreshAll,2000);
</script>
</head>
<body>
<h1>🌱 Smart Farming Dashboard</h1>

<h2>Mode: <span id="mode">MANUAL</span></h2>
<button class="btn" onclick="setMode('manual')">Manual</button>
<button class="btn" onclick="setMode('auto')">Otomatis</button>

<h2>Sensor Data</h2>
<div id="sensors" class="row"></div>

<h2>Relay Control</h2>
<div class="row">
<button class="btn" onclick="relay(1,'ON')">Relay1 ON</button>
<button class="btn" onclick="relay(1,'OFF')">Relay1 OFF</button>
<button class="btn" onclick="relay(2,'ON')">Relay2 ON</button>
<button class="btn" onclick="relay(2,'OFF')">Relay2 OFF</button>
<button class="btn" onclick="relay(3,'ON')">Relay3 ON</button>
<button class="btn" onclick="relay(3,'OFF')">Relay3 OFF</button>
</div>

<h2>Decision Tree Insight</h2>
<div id="decision"></div>

</body>
</html>
"""

@app.route("/")
def index(): return PAGE
@app.route("/sensors")
def sensors(): return jsonify({"values":sensor_data,"timestamps":sensor_timestamp,"relay":relay_state,"mode":control_mode["mode"]})
@app.route("/relay/<int:rid>/<state>", methods=["POST"])
def relay_route(rid,state):
    publish_relay(rid,state)
    return "OK"
@app.route("/mode/<m>", methods=["POST"])
def mode_route(m):
    control_mode["mode"] = m
    return "OK"
@app.route("/decision")
def decision_route():
    return jsonify(decision_info)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
