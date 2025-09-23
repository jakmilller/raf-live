/*
 * Emergency Stop Button for Kinova Robot
 * Arduino Micro USB Emergency Stop System
 *
 * Hardware:
 * - Arduino Micro
 * - Emergency stop button connected between pin 2 and GND
 * - Optional: LED connected to pin 13 for status indication
 *
 * When button is pressed, sends immediate "ESTOP" command via Serial
 */

const int ESTOP_BUTTON_PIN = 2;  // Digital pin 2 (interrupt capable)
const int STATUS_LED_PIN = 13;   // Built-in LED for status

volatile bool estopPressed = false;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // 50ms debounce
bool lastButtonState = HIGH;
bool buttonState = HIGH;

void setup() {
  // Initialize serial communication at high baud rate for fast response
  Serial.begin(115200);

  // Configure button pin with internal pull-up
  pinMode(ESTOP_BUTTON_PIN, INPUT_PULLUP);

  // Configure status LED
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  // Attach interrupt for immediate response
  attachInterrupt(digitalPinToInterrupt(ESTOP_BUTTON_PIN), estopInterrupt, FALLING);

  // Send ready signal
  Serial.println("ARDUINO_ESTOP_READY");
  digitalWrite(STATUS_LED_PIN, HIGH);  // LED on = ready
}

void loop() {
  // Handle debounced button reading in main loop
  int reading = digitalRead(ESTOP_BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;

      // Button pressed (LOW due to pull-up)
      if (buttonState == LOW) {
        sendEstopSignal();
      }
    }
  }

  lastButtonState = reading;

  // Blink LED to show system is alive
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink > 1000) {  // Blink every second
    digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
    lastBlink = millis();
  }

  delay(10);  // Small delay to prevent excessive CPU usage
}

// Interrupt service routine for immediate response
void estopInterrupt() {
  estopPressed = true;
  sendEstopSignal();
}

void sendEstopSignal() {
  // Send emergency stop signal immediately
  Serial.println("ESTOP");
  Serial.flush();  // Ensure immediate transmission

  // Flash LED rapidly to indicate emergency stop
  for (int i = 0; i < 10; i++) {
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(50);
    digitalWrite(STATUS_LED_PIN, LOW);
    delay(50);
  }
}