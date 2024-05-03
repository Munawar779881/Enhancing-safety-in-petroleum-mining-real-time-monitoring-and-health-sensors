#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
int pulsePin = A0;
int blinkPin = 13;

void setup() {
  pinMode(blinkPin, OUTPUT);
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
}

void loop() {
  int sensorValue = analogRead(pulsePin);

  Serial.print("Sensor Value: ");
  Serial.println(sensorValue);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Sensor Value: ");
  lcd.print(sensorValue);
  
  if (sensorValue < 200) {
    lcd.setCursor(0, 1);
    lcd.print("Pulse too low");
  }
}
