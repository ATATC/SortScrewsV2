#include <Servo.h>

Servo myServo;

const int SERVO_PIN = 9;
const int LED_PIN = 7;
int currentAngle = 0;

void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(0);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("Ready.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.equalsIgnoreCase("reset")) {
      currentAngle = 0;
      myServo.write(currentAngle);
      Serial.println("Servo reset to 0 degrees.");
      digitalWrite(13, HIGH);
    }
    else if (cmd.startsWith("turn ")) {
      String anglePart = cmd.substring(5);
      int angle = anglePart.toInt();

      if (angle < 0) angle = 0;
      if (angle > 180) angle = 180;

      currentAngle = angle;
      myServo.write(currentAngle);

      Serial.print("Servo moved to ");
      Serial.print(currentAngle);
      Serial.println(" degrees.");
      digitalWrite(LED_PIN, HIGH);
    }
    else {
      Serial.println("Unknown command. Use 'turn <degrees>' or 'reset'.");
      digitalWrite(LED_PIN, LOW);
    }
  }
}