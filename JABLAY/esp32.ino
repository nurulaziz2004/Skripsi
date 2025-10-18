#include <WiFi.h>
#include <PubSubClient.h>
#include "DHT.h"
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);



// ====== GPIO Relay ======
#define RELAY1 13
#define RELAY2 14
#define RELAY3 27
// pin sensor
int pin_dht11 = 5;
int pin_ldr = 33;
int pin_kelembaban_tanah1 = 34;
int pin_kelembaban_tanah2 = 35;
int pin_kelembaban_tanah3 = 32;

#define DHTTYPE DHT11  // Tipe sensor DHT11
DHT dht(pin_dht11, DHTTYPE);

// nilai sensor
float suhu = 0;
float kelembaban = 0;
float ldr = 0;
float kelembaban_tanah1 = 0;
float kelembaban_tanah2 = 0;
float kelembaban_tanah3 = 0;

int kondisi_p1 = 0;
int kondisi_p2 = 0;
int kondisi_p3 = 0;



// ====== Konfigurasi WiFi ======
const char* ssid = "wifi-iot";
const char* password = "password-iot";

// ====== Konfigurasi MQTT ======
const char* mqttServer = "103.186.1.210";
const int mqttPort = 1883;
WiFiClient espClient;
PubSubClient client(espClient);

// ====== Topik MQTT ======
String topicBase = "SatriaSensors773546";

String ldrPath = topicBase + "/ldr";
String tmpPath = topicBase + "/suhu";
String humPath = topicBase + "/kelembaban";
String humtPath1 = topicBase + "/kelembaban_tanah_1";
String humtPath2 = topicBase + "/kelembaban_tanah_2";
String humtPath3 = topicBase + "/kelembaban_tanah_3";

String relay1Path = topicBase + "/relay1";
String relay2Path = topicBase + "/relay2";
String relay3Path = topicBase + "/relay3";


void lcd_i2c(String text = "", int kolom = 0, int baris = 0) {
  byte bar[8] = {
    B11111,
    B11111,
    B11111,
    B11111,
    B11111,
    B11111,
    B11111,
  };
  if (text == "") {
    lcd.init();  //jika error pakai lcd.init();
    lcd.backlight();
    lcd.createChar(0, bar);
    lcd.setCursor(0, 0);
    lcd.print("Loading..");
    for (int i = 0; i < 16; i++) {
      lcd.setCursor(i, 1);
      lcd.write(byte(0));
      delay(100);
    }
    delay(50);
    lcd.clear();
  } else {
    lcd.setCursor(kolom, baris);
    lcd.print(text + "                ");
  }
}



void debug(String message, int row = 0, int clear = 1) {
  Serial.println(message);
  //tampilkan jika menggunakan lcd
  if (clear == 1) {
    lcd.clear();
  }
  lcd.setCursor(0, row);
  lcd.print(message);
}


// ====== Fungsi koneksi WiFi ======
void setupWiFi() {
  debug("Connecting to ");
  debug(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  debug("\nWiFi connected");
  delay(2000);
  Serial.print("IP address: ");
  debug(WiFi.localIP().toString());
  delay(2000);
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
      digitalWrite(RELAY1, LOW);
      Serial.println("Relay 1 ON");
      kondisi_p1 = 1;
    } else if (message == "OFF") {
      digitalWrite(RELAY1, HIGH);
      Serial.println("Relay 1 OFF");
      kondisi_p1 = 0;
    }
  }

  // Relay2
  if (String(topic) == relay2Path) {
    if (message == "ON") {
      digitalWrite(RELAY2, LOW);
      Serial.println("Relay 2 ON");
      kondisi_p2 = 1;
    } else if (message == "OFF") {
      digitalWrite(RELAY2, HIGH);
      Serial.println("Relay 2 OFF");
      kondisi_p2 = 0;
    }
  }

  // Relay3
  if (String(topic) == relay3Path) {
    if (message == "ON") {
      digitalWrite(RELAY3, LOW);
      Serial.println("Relay 3 ON");
      kondisi_p3 = 1;
    } else if (message == "OFF") {
      digitalWrite(RELAY3, HIGH);
      Serial.println("Relay 3 OFF");
      kondisi_p3 = 0;
    }
  }
}

// ====== Koneksi ke MQTT Broker ======
void reconnect() {
  while (!client.connected()) {
    debug("Attempting MQTT connection...");
    if (client.connect("SatriaSensors_ESP32")) {
      debug("MQTT connected");
      

      // subscribe semua topik relay
      client.subscribe(relay1Path.c_str());
      client.subscribe(relay2Path.c_str());
      client.subscribe(relay3Path.c_str());

      // test
      client.publish("esp/test", "Hello from ESP32");
      client.subscribe("esp/test");
      delay(1000);
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
  lcd_i2c();
  setupWiFi();
  client.setServer(mqttServer, mqttPort);
  client.setCallback(MQTTcallback);
  dht.begin();


  //pin sensor
  pinMode(pin_ldr, INPUT);
  pinMode(pin_kelembaban_tanah1, INPUT);
  pinMode(pin_kelembaban_tanah2, INPUT);
  pinMode(pin_kelembaban_tanah3, INPUT);


  // Relay sebagai output
  pinMode(RELAY1, OUTPUT);
  pinMode(RELAY2, OUTPUT);
  pinMode(RELAY3, OUTPUT);

  // default relay mati
  digitalWrite(RELAY1, HIGH);
  digitalWrite(RELAY2, HIGH);
  digitalWrite(RELAY3, HIGH);
}

void baca_sensor() {
  kelembaban = dht.readHumidity();
  suhu = dht.readTemperature();  // Celcius
  ldr = constrain(100 - map(analogRead(pin_ldr), 0, 4095, 0, 100), 0, 100);
  kelembaban_tanah1 = constrain(100 - map(analogRead(pin_kelembaban_tanah1), 2400, 4095, 0, 100), 0, 100);
  kelembaban_tanah2 = constrain(100 - map(analogRead(pin_kelembaban_tanah2), 2400, 4095, 0, 100), 0, 100);
  kelembaban_tanah3 = constrain(100 - map(analogRead(pin_kelembaban_tanah3), 2000, 4095, 0, 100), 0, 100);

  Serial.print("Data sensor suhu : " + String(suhu) + "/kelembaban : " + String(kelembaban) + "/ldr : " + String(ldr));
  Serial.println("/kt1 : " + String(kelembaban_tanah1) + "/kt2 : " + String(kelembaban_tanah2) + "/kt3 : " + String(kelembaban_tanah3));
}

int timer_tampilan = 3000;
int timer1 = 0;
int mode = 0;

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  // Baca sensor
  baca_sensor();

  //send_mosquito(ldrPath, random(100, 900));
  send_mosquito(ldrPath, ldr);
  send_mosquito(tmpPath, suhu);
  send_mosquito(humPath, kelembaban);
  send_mosquito(humtPath1, kelembaban_tanah1);
  send_mosquito(humtPath2, kelembaban_tanah2);
  send_mosquito(humtPath3, kelembaban_tanah3);

  Serial.println("Dummy data sent...");
  


  if (millis() - timer1 >= timer_tampilan) {
    mode += 1;
    timer1 = millis();
  }
  if (mode >= 3) {
    mode = 0;
  }

  if (mode == 0) {
    debug("S:" + String(suhu) + " K:" + String(kelembaban));
    debug("ldr: " + String(ldr), 1, 0);
  } else if (mode == 1) {
    debug("KT1:" + String(kelembaban_tanah1));
    debug("KT2:" + String(kelembaban_tanah2), 1, 0);
  } else if (mode == 2) {

    debug("KT3:" + String(kelembaban_tanah3) );
    debug("P1:" + String(kondisi_p1) + " P2:" + String(kondisi_p2) + " P3:" + String(kondisi_p3), 1, 0);
  }
  delay(10);
}
