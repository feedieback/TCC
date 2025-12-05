#include <WiFi.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
#include <DHT.h>

// ---------------- CONFIG WIFI ----------------
#define WIFI_SSID     "xxxxxxxxxx"
#define WIFI_PASS     "xxxxxxxxxx"    

// ---------------- CONFIG ADAFRUIT IO ----------------
#define AIO_SERVER      "io.adafruit.com"
#define AIO_SERVERPORT  1883
#define AIO_USERNAME    "xxxxxxxxxx"
#define AIO_KEY         "xxxxxxxxxx"

// ---------------- CONFIG SENSORES ----------------
#define DHTPIN   4
#define DHTTYPE  DHT11
DHT dht(DHTPIN, DHTTYPE);

#define LDR_PIN   34      // Entrada analógica
#define SOLO_PIN  35      // Sensor de umidade do solo (A0 do módulo)

// Valores de calibração do sensor resistivo (AJUSTE SE PRECISAR)
int solo_molhado = 800;     // valor RAW quando o solo está bem úmido
int solo_seco    = 3500;    // valor RAW quando o solo está totalmente seco

// ---------------- MQTT CLIENT ----------------
WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USERNAME, AIO_KEY);

// Feeds MQTT
Adafruit_MQTT_Publish feedTemp       = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/temperatura");
Adafruit_MQTT_Publish feedUmidadeAr  = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/umidade-ar");
Adafruit_MQTT_Publish feedUmidSolo   = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/umidade-solo");
Adafruit_MQTT_Publish feedLuz        = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/indice-luz");


// ---------------- FUNÇÃO: RECONEXÃO MQTT ----------------
void MQTT_connect() {
  int8_t ret;

  if (mqtt.connected()) return;

  Serial.print("Conectando ao MQTT... ");

  while ((ret = mqtt.connect()) != 0) {
    Serial.println(mqtt.connectErrorString(ret));
    Serial.println("Tentando novamente em 5s...");
    mqtt.disconnect();
    delay(5000);
  }

  Serial.println("MQTT Conectado!");
}


// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);
  delay(2000);

  dht.begin();
  pinMode(LDR_PIN, INPUT);
  pinMode(SOLO_PIN, INPUT);

  Serial.println("Conectando ao Wi-Fi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(400);
  }
  Serial.println("\nWi-Fi conectado!");
}


// ---------------- LOOP ----------------
void loop() {
  MQTT_connect();

  // ---- Leitura dos sensores ----
  float temperatura = dht.readTemperature();
  float umidade_ar = dht.readHumidity();
  int luz_raw      = analogRead(LDR_PIN);
  int solo_raw     = analogRead(SOLO_PIN);

  if (isnan(temperatura) || isnan(umidade_ar)) {
    Serial.println("Erro ao ler DHT11!");
    delay(2000);
    return;
  }

  // ---- Conversão do solo em porcentagem ----
  int solo_percent = map(solo_raw, solo_seco, solo_molhado, 0, 100);
  solo_percent = constrain(solo_percent, 0, 100);

  // ---- Conversão da luz ----
  int luz_percent = map(luz_raw, 0, 4095, 0, 100);

  // ---- Logs Serial ----
  Serial.println("===============================");
  Serial.printf("Temperatura: %.2f °C\n", temperatura);
  Serial.printf("Umidade do Ar: %.2f %%\n", umidade_ar);
  Serial.printf("Solo: %d %%  | RAW = %d\n", solo_percent, solo_raw);
  Serial.printf("Luminosidade: %d %% | RAW = %d\n", luz_percent, luz_raw);

  // ---- Envio MQTT ----
  feedTemp.publish(temperatura);
  feedUmidadeAr.publish(umidade_ar);
  feedUmidSolo.publish(solo_percent);
  feedLuz.publish(luz_percent);

  Serial.println("Dados enviados para Adafruit IO!");

  delay(7000);  // intervalo de envio (7s)
}
