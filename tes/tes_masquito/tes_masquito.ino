#include <WiFi.h>
#include <PubSubClient.h>

// ====== Konfigurasi WiFi ======
const char* ssid = "wifi-iot";
const char* password = "password-iot";

// ====== Konfigurasi MQTT ======
const char* mqttServer = "test.mosquitto.org";
const int mqttPort = 1883;
WiFiClient espClient;
PubSubClient client(espClient);

// ====== Topik MQTT ======
String topicBase = "SatriaSensors773546";

String ldrPath   = topicBase + "/ldr";
String tmpPath   = topicBase + "/suhu";
String humPath   = topicBase + "/kelembaban";
String humtPath1 = topicBase + "/kelembaban_tanah_1";
String humtPath2 = topicBase + "/kelembaban_tanah_2";
String humtPath3 = topicBase + "/kelembaban_tanah_3";

String relay1Path = topicBase + "/relay1";
String relay2Path = topicBase + "/relay2";
String relay3Path = topicBase + "/relay3";

// ====== GPIO Relay ======
#define RELAY1 25
#define RELAY2 26
#define RELAY3 27

// ====== Fungsi koneksi WiFi ======
void setupWiFi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// ====== Callback saat ada pesan MQTT ======
void MQTTcallback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("Message [");
  Serial.print(topic);
  Serial.print("] = ");
  Serial.println(message);

  // Relay1
  if (String(topic) == relay1Path) {
    if (message == "ON") {
      digitalWrite(RELAY1, HIGH);
      Serial.println("Relay 1 ON");
    } else if (message == "OFF") {
      digitalWrite(RELAY1, LOW);
      Serial.println("Relay 1 OFF");
    }
  }

  // Relay2
  if (String(topic) == relay2Path) {
    if (message == "ON") {
      digitalWrite(RELAY2, HIGH);
      Serial.println("Relay 2 ON");
    } else if (message == "OFF") {
      digitalWrite(RELAY2, LOW);
      Serial.println("Relay 2 OFF");
    }
  }

  // Relay3
  if (String(topic) == relay3Path) {
    if (message == "ON") {
      digitalWrite(RELAY3, HIGH);
      Serial.println("Relay 3 ON");
    } else if (message == "OFF") {
      digitalWrite(RELAY3, LOW);
      Serial.println("Relay 3 OFF");
    }
  }
}

// ====== Koneksi ke MQTT Broker ======
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("SatriaSensors_ESP32")) {
      Serial.println("connected");

      // subscribe semua topik relay
      client.subscribe(relay1Path.c_str());
      client.subscribe(relay2Path.c_str());
      client.subscribe(relay3Path.c_str());

      // test
      client.publish("esp/test", "Hello from ESP32");
      client.subscribe("esp/test");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// ====== Kirim data MQTT ======
void send_mosquito(String path, float pesan) {
  client.publish(path.c_str(), String(pesan).c_str());
}

void setup() {
  Serial.begin(115200);
  setupWiFi();
  client.setServer(mqttServer, mqttPort);
  client.setCallback(MQTTcallback);

  // Relay sebagai output
  pinMode(RELAY1, OUTPUT);
  pinMode(RELAY2, OUTPUT);
  pinMode(RELAY3, OUTPUT);

  // default relay mati
  digitalWrite(RELAY1, LOW);
  digitalWrite(RELAY2, LOW);
  digitalWrite(RELAY3, LOW);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // kirim dummy data tiap 2 detik
  send_mosquito(ldrPath, random(100, 900));
  send_mosquito(tmpPath, random(20, 35));
  send_mosquito(humPath, random(40, 80));
  send_mosquito(humtPath1, random(20, 60));
  send_mosquito(humtPath2, random(20, 60));
  send_mosquito(humtPath3, random(20, 60));

  Serial.println("Dummy data sent...");
  delay(100);
}
