#include <Servo.h>

Servo servoA;
Servo servoB;

const int SERVO_A_PIN = 9;
const int SERVO_B_PIN = 10;
int MIN_ANGLE_A = 77;
int MAX_ANGLE_A = 170;
int MIN_ANGLE_B = 70;
int MAX_ANGLE_B = 160;
int currentAngleA = MIN_ANGLE_A;
int currentAngleB = MIN_ANGLE_B;

void setup() {
  Serial.begin(9600);
  servoA.attach(SERVO_A_PIN);
  servoA.write(MIN_ANGLE_A);
  servoB.attach(SERVO_B_PIN);
  servoB.write(MIN_ANGLE_B);
  Serial.println("Ready.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.equalsIgnoreCase("reset a")) {
      currentAngleA = MIN_ANGLE_A;
      servoA.write(currentAngleA);
      Serial.print("Servo A reset to ");
      Serial.print(currentAngleA);
      Serial.println(" degrees.");
    }
    else if (cmd.equalsIgnoreCase("reset b")) {
      currentAngleB = MIN_ANGLE_B;
      servoB.write(currentAngleB);
      Serial.print("Servo B reset to ");
      Serial.print(currentAngleB);
      Serial.println(" degrees.");
    }
    else if (cmd.startsWith("turn A ")) {
      String anglePart = cmd.substring(7);
      int angle = anglePart.toInt();

      if (angle < MIN_ANGLE_A) angle = MIN_ANGLE_A;
      if (angle > MAX_ANGLE_A) angle = MAX_ANGLE_A;

      currentAngleA = angle;
      servoA.write(currentAngleA);

      Serial.print("Servo A moved to ");
      Serial.print(currentAngleA);
      Serial.println(" degrees.");
    }
    else if (cmd.startsWith("turn B ")) {
      String anglePart = cmd.substring(7);
      int angle = anglePart.toInt();

      if (angle < MIN_ANGLE_B) angle = MIN_ANGLE_B;
      if (angle > MAX_ANGLE_B) angle = MAX_ANGLE_B;

      currentAngleB = angle;
      servoB.write(currentAngleB);

      Serial.print("Servo B moved to ");
      Serial.print(currentAngleB);
      Serial.println(" degrees.");
    }
    else {
      Serial.println("Unknown command. Use 'turn <degrees>' or 'reset'.");
    }
  }
}