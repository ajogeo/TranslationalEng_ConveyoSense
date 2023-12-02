#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <WiFi.h>
#include <WebSocketsClient.h>  // Include the WebSocket client library

const char* ssid = "Ajo-Redmi";      // Replace with your network SSID
const char* password = "8971471834";  // Replace with your network password
const char* host = "192.168.153.202";   // Replace with the IP address of your WebSocket server
const int port = 8765;  // Replace with the port your server is running on

const int microphonePin = 34;
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

unsigned long previousMillis = 0;
const long interval = 1;  // Adjust the interval as needed (in milliseconds)

// Initialize the WebSocket client
WebSocketsClient webSocket;

void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("WebSocket connected");
      break;
    case WStype_DISCONNECTED:
      Serial.println("WebSocket disconnected");
      break;
  }
}

void setup(void) {
  Serial.begin(115200);
  Serial.println("Accelerometer Test");
  Serial.println("");

  if (!accel.begin()) {
    Serial.println("Ooops, no ADXL345 detected ... Check your wiring!");
    while (1);
  }

  accel.setRange(ADXL345_RANGE_16_G);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Initialize WebSocket
  webSocket.begin(host, port, "/");
  webSocket.onEvent(onWebSocketEvent);
}

void loop(void) {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    sensors_event_t event;
    accel.getEvent(&event);

    float X = event.acceleration.x;
    float Y = event.acceleration.y;
    float Z = event.acceleration.z;

    int microphoneData = analogRead(microphonePin);

    // Print accelerometer and microphone data to the serial port
    Serial.print("Accelerometer - X: "); Serial.print(X); Serial.print("  Y: "); Serial.print(Y); Serial.print(" Z: "); Serial.println(Z);
    Serial.print("Microphone Data: "); Serial.println(microphoneData);

    // Send data to the WebSocket server
    sendSensorData(X, Y, Z, microphoneData);
  }

  // Handle WebSocket events
  webSocket.loop();
}

void sendSensorData(float accelX, float accelY, float accelZ, int microphoneData) {
  // Create a JSON object to send data
  String jsonPayload = "{\"accelX\":" + String(accelX) + ",";
  jsonPayload += "\"accelY\":" + String(accelY) + ",";
  jsonPayload += "\"accelZ\":" + String(accelZ) + ",";
  jsonPayload += "\"microphoneData\":" + String(microphoneData) + "}";

  // Send the JSON data to the WebSocket server
  webSocket.sendTXT(jsonPayload);
}
