# ====== Install ======
# pip install paho-mqtt==2.1.0 flask==3.1.1 scikit-learn==1.5.2 matplotlib==3.9.2 pandas==2.2.3

import os
import time
import threading
from datetime import datetime
import psycopg2
import redis
import logging
# ===================== Konfigurasi REDIS =====================
REDIS_CONFIG = {
    "host": "redis.raishannan.com",
    "port": 63802,
    "username": "redis",
    "password": "CrGyLcQ3HSonzLUYy38oXHlylNcTzEE5Q9Jq0h9HhY1gYwxLOMyvhQfCksLHWVUL"
}

redis_client = redis.Redis(
    host=REDIS_CONFIG["host"],
    port=REDIS_CONFIG["port"],
    password=REDIS_CONFIG["password"],
    username=REDIS_CONFIG["username"],
    decode_responses=True
)

# Configure logging: default WARNING to keep output quiet. Use LOG_LEVEL env to override.
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
numeric_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
# Silence Flask/Werkzeug access logs (the repeated GET /... lines)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('werkzeug.serving').setLevel(logging.WARNING)

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
    matplotlib.use('Agg')  # gunakan backend non-GUI
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
BROKER = "mosquitto.raishannan.com"
PASSWORD = "Mq0KuqbH1TU231U36o7nLSMwrscil6bg"
USER = "admin"
PORT = 1883
TOPIC_BASE = "SatriaSensors773546"
current_folder = os.path.dirname(os.path.abspath(__file__))
print("current FOLDER : ",current_folder)

# ===================== Konfigurasi DB =====================
DB_CONFIG = {
    "host": "postgre.raishannan.com",
    "port": 54321,
    "user": "jablai_user",
    "password": "9pTsQwHrVdFkNgLmZcXbJvKsLaPiOtUeYhRxDoWmCnEsQfGaBjVlKhNcTzPiUyW",
    "dbname": "jablai",
    "sslmode": "disable"
}

def get_db_conn():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dbname=DB_CONFIG["dbname"],
        sslmode=DB_CONFIG["sslmode"]
    )


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

# ===================== State Global =====================
sensor_data = {k: None for k in SENSOR_KEYS}
sensor_timestamp = {k: None for k in SENSOR_KEYS}
control_mode = {"mode": "manual"}
relay_state = {1: "OFF", 2: "OFF", 3: "OFF"}
manual_override = {}  # relay_id -> datetime of last manual command
MANUAL_OVERRIDE_TTL = 120  # seconds to respect manual override before auto can take over
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
    print("Connected with result code", rc)
    for t in sensor_topics:
        client.subscribe(t)
    for t in RELAY_TOPICS.values():
        client.subscribe(t)

def safe_float(s):
    try:
        return float(str(s).strip())
    except:
        return None

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")
    key = topic.split("/")[-1]

    if key in sensor_data:
        val = safe_float(payload)
        sensor_data[key] = val
        sensor_timestamp[key] = datetime.now().strftime("%H:%M:%S")
        # Buffer ke Redis
        try:
            now = int(time.time())
            redis_key = f"sensor_buffer:{now}"
            redis_client.hset(redis_key, key, val)
            redis_client.expire(redis_key, 2)  # auto-expire
            # Cek jika sudah lengkap 6 data
            if redis_client.hlen(redis_key) == 6:
                data = redis_client.hgetall(redis_key)
                try:
                    conn = get_db_conn()
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO sensor (suhu, kelembaban, kelembaban_tanah1, kelembaban_tanah2, kelembaban_tanah3, ldr, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data.get("suhu"),
                        data.get("kelembaban"),
                        data.get("kelembaban_tanah_1"),
                        data.get("kelembaban_tanah_2"),
                        data.get("kelembaban_tanah_3"),
                        data.get("ldr"),
                        datetime.now(),
                        datetime.now()
                    ))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    print("[DB SENSOR ERROR]", e)
                redis_client.delete(redis_key)
        except Exception as e:
            print("[REDIS SENSOR ERROR]", e)
    for rid, rtopic in RELAY_TOPICS.items():
        if topic == rtopic:
            print(f"[RELAY-ECHO] relay{rid} <- {payload}")
            # Update status relay di database kontrol berdasarkan feedback ESP32
            try:
                conn = get_db_conn()
                cur = conn.cursor()
                cur.execute("UPDATE kontrol SET status=%s, updated_at=%s WHERE name=%s", (
                    payload,
                    datetime.now(),
                    f"pompa{rid}"
                ))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print("[DB RELAY FEEDBACK ERROR]", e)

client = mqtt.Client(client_id="SatriaSensors_FlaskDT", protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
# set username/password if provided
try:
    client.username_pw_set(USER, PASSWORD)
except Exception:
    logging.debug("Could not set MQTT username/password")
# Use non-blocking connect so the process doesn't crash if broker is down; paho will handle reconnects
try:
    # prefer connect_async which returns immediately and reconnects in background
    client.connect_async(BROKER, PORT, 60)
except Exception as e:
    print("[MQTT CONNECT_ASYNC ERROR]", e)
    # fallback to a safe, wrapped connect attempt (won't raise here because we catch exceptions)
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e2:
        print("[MQTT CONNECT ERROR]", e2)
client.loop_start()

def publish_relay(relay_id: int, state: str):
    state = "ON" if str(state).upper() == "ON" else "OFF"
    topic = RELAY_TOPICS.get(relay_id)
    if topic:
        try:
            info = client.publish(topic, state, qos=1)
            try:
                info.wait_for_publish(timeout=0.5)
            except Exception:
                pass
        except Exception:
            # fallback to simple publish
            client.publish(topic, state)
        relay_state[relay_id] = state
        print(f"[RELAY] relay{relay_id} => {state}")
        # Update status relay di database kontrol
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("UPDATE kontrol SET status=%s, updated_at=%s WHERE name=%s", (
                state,
                datetime.now(),
                f"pompa{relay_id}"
            ))
            conn.commit()
            cur.close()
            conn.close()
            # mark manual override time when this publish originates from a manual path
            try:
                manual_override[relay_id] = datetime.now()
            except Exception:
                pass
        except Exception as e:
            print("[DB RELAY ERROR]", e)

def fast_publish_thread(relay_id: int, state: str):
    """Background fast publish: publish via MQTT client with QoS=1 and retries, update in-memory state (no DB write)."""
    topic = RELAY_TOPICS.get(relay_id)
    if not topic:
        return
    attempts = 0
    max_attempts = 4
    while attempts < max_attempts:
        attempts += 1
        try:
            info = client.publish(topic, state, qos=1)
            # wait for publish to complete (MQTTMessageInfo.wait_for_publish)
            try:
                # some paho versions accept timeout, others don't
                info.wait_for_publish(timeout=1)
            except TypeError:
                try:
                    info.wait_for_publish()
                except Exception:
                    pass
            # check if published (is_published may exist)
            published = True
            try:
                if hasattr(info, 'is_published'):
                    published = info.is_published()
            except Exception:
                published = True
            if published:
                relay_state[relay_id] = state
                print(f"[FAST-RELAY] relay{relay_id} => {state} (attempt {attempts})")
                # persist to DB so manual override is respected
                try:
                    conn = get_db_conn()
                    cur = conn.cursor()
                    cur.execute("UPDATE kontrol SET status=%s, updated_at=%s WHERE name=%s", (
                        state,
                        datetime.now(),
                        f"pompa{relay_id}"
                    ))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    print("[DB FAST-RELAY UPDATE ERROR]", e)
                try:
                    manual_override[relay_id] = datetime.now()
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[FAST-RELAY ERROR] attempt {attempts}", e)
            try:
                client.reconnect()
            except Exception:
                time.sleep(0.05)
    print(f"[FAST-RELAY FAILED] relay{relay_id} => {state} after {max_attempts} attempts")

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
                        # respect manual override if recent
                        mo = manual_override.get(i)
                        if mo is not None and (datetime.now() - mo).total_seconds() < MANUAL_OVERRIDE_TTL:
                            # skip automatic control for this pot
                            continue
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
<script src="https://unpkg.com/mqtt/dist/mqtt.min.js"></script>
<script>
// Robust MQTT-over-WebSocket client for browser
let wsClient = null;
const BROKER_WS = 'ws://103.186.1.210:8083/mqtt'; // change if your broker uses different WS port/path
const TOPIC_BASE_JS = "{{ TOPIC_BASE }}";
let pubQueue = []; // queued publishes when offline
const MAX_QUEUE_ATTEMPTS = 5;

function initBrowserMQTT(){
    try{
        wsClient = mqtt.connect(BROKER_WS, {connectTimeout:4000, reconnectPeriod:1000, clean:true});
        wsClient.on('connect', ()=>{
            console.log('MQTT WS connected');
            // subscribe to relay feedback topics
            ['relay1','relay2','relay3'].forEach(t=> wsClient.subscribe(`${TOPIC_BASE_JS}/${t}`, {qos:1}, (err)=>{ if(err) console.log('sub err',err)}));
            // flush any queued publishes
            flushQueue();
        });
        wsClient.on('reconnect', ()=> console.log('MQTT WS reconnecting'));
        wsClient.on('offline', ()=> console.log('MQTT WS offline'));
        wsClient.on('error', (e)=> console.log('MQTT WS error', e));

        wsClient.on('message', (topic, message) => {
            try{
                const key = topic.split('/').pop();
                const payload = message.toString();
                if(key.startsWith('relay')){
                    const rid = key.replace('relay','');
                    document.getElementById('r'+rid).innerText = payload;
                    document.getElementById('r'+rid+'_on').classList.toggle('active', payload=='ON');
                    document.getElementById('r'+rid+'_off').classList.toggle('active', payload=='OFF');
                }
            }catch(e){ console.log('ws msg err', e); }
        });
    }catch(e){ console.log('initBrowserMQTT error', e); }
}

function enqueuePublish(topic, msg, qos=1){
    pubQueue.push({topic, msg, qos, attempts:0});
}

function flushQueue(){
    if(!wsClient || !wsClient.connected) return;
    const pending = pubQueue.slice();
    pubQueue = [];
    pending.forEach(item=>{
        wsClient.publish(item.topic, item.msg, {qos: item.qos}, (err)=>{
            if(err){
                item.attempts = (item.attempts||0) + 1;
                if(item.attempts < MAX_QUEUE_ATTEMPTS) pubQueue.push(item);
            }
        });
    });
}

// periodically try flushing queue
setInterval(()=>{ try{ flushQueue(); }catch(e){} }, 800);

async function setMode(mode){ await fetch('/mode/'+mode, {method:'POST'}); refreshAll(); }

async function publishRelayDirect(rid, state){
    state = (String(state).toUpperCase()==='ON') ? 'ON' : 'OFF';
    const topic = `${TOPIC_BASE_JS}/relay${rid}`;
    // First: call synchronous server publish (wait for confirmation)
    try{
        const res = await fetch(`/fast_relay_sync/${rid}/${state}`, {method:'POST'});
        if(res && res.ok){
            // we got server ack — optimistic UI update
            document.getElementById('r'+rid).innerText = state;
            document.getElementById('r'+rid+'_on').classList.toggle('active', state=='ON');
            document.getElementById('r'+rid+'_off').classList.toggle('active', state=='OFF');
        }
    }catch(e){
        // server sync failed — fallback to fire+forget fast endpoint and WS enqueue
        fetch(`/fast_relay/${rid}/${state}`, {method:'POST'}).catch(()=>{});
    }
    // Also attempt WS publish for redundancy/instant UI feedback
    if(wsClient && wsClient.connected){
        wsClient.publish(topic, state, {qos:1}, (err)=>{ if(err){ console.log('ws publish err', err); enqueuePublish(topic,state,1); } });
    }else{
        enqueuePublish(topic, state, 1);
    }
}

async function relay(rid,state){ publishRelayDirect(rid,state); }

window.addEventListener('load', ()=>{ initBrowserMQTT(); });

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
        if(!(wsClient && wsClient.connected)){
            // only update from server state if WS not present (WS will update UI on its own)
            document.getElementById('r'+rid).innerText=state;
            document.getElementById('r'+rid+'_on').classList.toggle('active', state=='ON');
            document.getElementById('r'+rid+'_off').classList.toggle('active', state=='OFF');
        }
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
async function refreshAll(){ refreshSensors(); refreshDecision(); refreshRules(); }
setInterval(refreshAll,500); window.onload=refreshAll;
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
def index():
    # render PAGE with TOPIC_BASE injected for JS
    return render_template_string(PAGE, TOPIC_BASE=TOPIC_BASE)

@app.route("/mode/<mode>", methods=["POST"])
def set_mode(mode):
    control_mode["mode"] = "auto" if mode.lower().strip()=="auto" else "manual"
    # if switching to auto, clear manual overrides so auto can immediately take over
    if control_mode["mode"] == "auto":
        manual_override.clear()
    return jsonify(ok=True, mode=control_mode["mode"])

@app.route("/relay/<int:rid>/<state>", methods=["POST"])
def set_relay(rid,state):
    publish_relay(rid,state)
    return jsonify(ok=True, relay=rid, state=relay_state[rid])


@app.route('/fast_relay/<int:rid>/<state>', methods=['POST'])
def fast_relay(rid, state):
    """Immediate publish endpoint: spawn a background thread to publish without DB update for minimal latency."""
    state_norm = 'ON' if str(state).upper() == 'ON' else 'OFF'
    threading.Thread(target=fast_publish_thread, args=(rid, state_norm), daemon=True).start()
    # optimistic immediate response
    return jsonify(ok=True, relay=rid, state=state_norm, method='fast')


@app.route('/fast_relay_sync/<int:rid>/<state>', methods=['POST'])
def fast_relay_sync(rid, state):
    """Synchronous publish endpoint: try to publish with QoS=1 and wait for confirmation before returning."""
    state_norm = 'ON' if str(state).upper() == 'ON' else 'OFF'
    topic = RELAY_TOPICS.get(rid)
    if not topic:
        return jsonify(ok=False, error='unknown topic'), 400
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        attempts += 1
        try:
            info = client.publish(topic, state_norm, qos=1)
            try:
                info.wait_for_publish(timeout=1)
            except Exception:
                pass
            # persist manual command in DB and in-memory
            try:
                conn = get_db_conn()
                cur = conn.cursor()
                cur.execute("UPDATE kontrol SET status=%s, updated_at=%s WHERE name=%s", (
                    state_norm,
                    datetime.now(),
                    f"pompa{rid}"
                ))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print("[DB FAST-RELAY_SYNC UPDATE ERROR]", e)
            relay_state[rid] = state_norm
            try:
                manual_override[rid] = datetime.now()
            except Exception:
                pass
            return jsonify(ok=True, relay=rid, state=state_norm, attempts=attempts)
        except Exception as e:
            try:
                client.reconnect()
            except Exception:
                time.sleep(0.05)
    return jsonify(ok=False, error='publish_failed'), 500

@app.route("/sensors")
def sensors_api():
    return jsonify({"values": sensor_data, "timestamps": sensor_timestamp,
                    "mode": control_mode["mode"], "relay": relay_state})

@app.route("/decision")
def decision_api(): return jsonify(decision_info)

@app.route("/rules")
def rules_api(): return tree_rules_text


@app.route('/health')
def health_check():
    report = {"redis": False, "db": False, "mqtt": False}
    # Redis
    try:
        report['redis'] = redis_client.ping()
    except Exception as e:
        report['redis_error'] = str(e)
    # Postgres
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.fetchone()
        cur.close()
        conn.close()
        report['db'] = True
    except Exception as e:
        report['db_error'] = str(e)
    # MQTT
    try:
        report['mqtt'] = getattr(client, 'is_connected', lambda: True)()
    except Exception as e:
        report['mqtt_error'] = str(e)
    return jsonify(report)

if __name__=="__main__":
    # enable threading so the server can handle fast_relay posts concurrently
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
