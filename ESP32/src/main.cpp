#include <ESP32Servo.h>

const int servoPin  = 5;  // Chan tin hieu PWM Servo MG996R (Dieu khien goc lai)
const int motorPin1 = 4;  // Chan IN1 Driver L298N (Tien)
const int motorPin2 = 2;  // Chan IN2 Driver L298N (Lui)

const int pwmFreq = 5000;
const int pwmResolution = 8; // 8-bit (0 -> 255)
const int pwmChannel1 = 4;
const int pwmChannel2 = 5;

const int SERVO_CENTER         = 90;  // Goc thang lai chuan (dieu chinh neu xe bi lech)
const int SERVO_MAX_DEFLECTION = 60;  // Do lech goc lai toi da 60 do (30 den 150 do -> xoay het co khi cua va khi lui)
const int SERVO_MIN_ANGLE      = 30;  // Gioi han goc lai toi da sang trai (30 do)
const int SERVO_MAX_ANGLE      = 150; // Gioi han goc lai toi da sang phai (150 do)
const bool SERVO_INVERT        = false; // false: +error (SHARP_R) quay phai, -error (SHARP_L) quay trai (doi true neu nguoc)

const int MIN_START_PWM        = 65;  
const int MIN_REVERSE_PWM      = 100; 
const unsigned long TIMEOUT_MS = 350; 

Servo myServo;
unsigned long lastPacketTime = 0;
bool isStopped = true;

char rxBuffer[48];
int rxIndex = 0;
bool recvInProgress = false;

void setMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);

  #if ESP_ARDUINO_VERSION_MAJOR >= 3
    if (speed > 0) {
      ledcWrite(motorPin1, speed);
      ledcWrite(motorPin2, 0);
    } else if (speed < 0) {
      ledcWrite(motorPin1, 0);
      ledcWrite(motorPin2, abs(speed));
    } else {
      ledcWrite(motorPin1, 0);
      ledcWrite(motorPin2, 0);
    }
  #else
    if (speed > 0) {
      ledcWrite(pwmChannel1, speed);
      ledcWrite(pwmChannel2, 0);
    } else if (speed < 0) {
      ledcWrite(pwmChannel1, 0);
      ledcWrite(pwmChannel2, abs(speed));
    } else {
      ledcWrite(pwmChannel1, 0);
      ledcWrite(pwmChannel2, 0);
    }
  #endif
}

// 5. HAM XU LY GOI TIN TU RASPBERRY PI (<SPEED,ERROR,CMD>)
//    - SPEED > 0: Tien (1..100%)
//    - SPEED < 0: Lui nhe phuc hoi khi mat vach (-1..-100%)
//    - SPEED == 0 hoac CMD == "STOP": Dung xe
static char* trimToken(char* s) {
  if (!s) return s;
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
  int len = strlen(s);
  while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\r' || s[len - 1] == '\n')) {
    s[--len] = '\0';
  }
  return s;
}

void parseAndExecute(char* packet) {
  // Tach SPEED (-100 .. 100)
  char* token = strtok(packet, ",");
  if (!token) return;
  int speed_pct = atoi(trimToken(token));

  // Tach ERROR (-1.00 .. +1.00)
  token = strtok(NULL, ",");
  if (!token) return;
  float lane_error = atof(trimToken(token));

  // Tach CMD (NORM | CURVE | SHARP_L | SHARP_R | REV_L | REV_R | SLOW | STOP)
  token = strtok(NULL, ",");
  if (!token) return;
  token = trimToken(token);
  char cmd[16];
  strncpy(cmd, token, sizeof(cmd) - 1);
  cmd[sizeof(cmd) - 1] = '\0';

  // Cap nhat thoi gian nhan tin hop le de reset Watchdog an toan
  lastPacketTime = millis();

  // A. XU LY LENH DUNG HAN (STOP / BIEN BAO STOP / DEN DO / MAT LAN QUA LAU)
  // Luu y: chi khi cmd == "STOP" moi tra thang servo; neu speed_pct == 0 voi cmd khac (vd: PRE_REV), van giu goc servo!
  if (strcmp(cmd, "STOP") == 0) {
    setMotorSpeed(0);
    myServo.write(SERVO_CENTER); // Tra thang lai khi dung han
    isStopped = true;
    return;
  }

  // B. XU LY GOC LAI CHO SERVO DUA VAO DO LECH LAN (ERROR: -1.0 -> +1.0)
  lane_error = constrain(lane_error, -1.0f, 1.0f);
  int steer_offset = (int)round(lane_error * SERVO_MAX_DEFLECTION);
  if (SERVO_INVERT) {
    steer_offset = -steer_offset;
  }

  // Uu tien goc cua gat 90 do hoac lenh lui quay servo (REV_L / REV_R)
  if (strcmp(cmd, "SHARP_L") == 0 || strcmp(cmd, "SIGN_L") == 0 || strcmp(cmd, "REV_L") == 0) {
    steer_offset = SERVO_INVERT ? SERVO_MAX_DEFLECTION : -SERVO_MAX_DEFLECTION;
  } else if (strcmp(cmd, "SHARP_R") == 0 || strcmp(cmd, "SIGN_R") == 0 || strcmp(cmd, "REV_R") == 0) {
    steer_offset = SERVO_INVERT ? -SERVO_MAX_DEFLECTION : SERVO_MAX_DEFLECTION;
  }

  int target_angle = constrain(SERVO_CENTER + steer_offset, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  myServo.write(target_angle);

  // C. XU LY TOC DO DONG CO DC (HO TRO CA TIEN > 0, DUNG GIU GOC == 0, VA LUI < 0)
  int target_pwm = 0;
  if (speed_pct > 0) {
    int clamped_spd = constrain(speed_pct, 1, 100);
    target_pwm = map(clamped_spd, 1, 100, MIN_START_PWM, 255);
  } else if (speed_pct < 0) {
    int abs_spd = constrain(abs(speed_pct), 1, 100);
    target_pwm = -map(abs_spd, 1, 100, MIN_REVERSE_PWM, 255);
  }
  setMotorSpeed(target_pwm);
  isStopped = (target_pwm == 0);
}

// -----------------------------------------------------------------------------
// 6. HAM DOC SERIAL KHONG CHAN (NON-BLOCKING)
// -----------------------------------------------------------------------------
void handleSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '<') {
      recvInProgress = true;
      rxIndex = 0;
      rxBuffer[0] = '\0';
    } else if (recvInProgress) {
      if (c == '>') {
        recvInProgress = false;
        rxBuffer[rxIndex] = '\0';
        parseAndExecute(rxBuffer);
      } else if (c == '\r' || c == '\n') {
        // Bo qua ky tu xuong dong
      } else {
        if (rxIndex < (int)(sizeof(rxBuffer) - 1)) {
          rxBuffer[rxIndex++] = c;
        } else {
          // Phong tranh tran bo dem
          recvInProgress = false;
          rxIndex = 0;
        }
      }
    }
  }
}
void setup() {
  Serial.begin(115200);
  
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // Khoi tao Servo tren chan 
  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, 500, 2400);

  // Khoi tao PWM dieu khien Motor L298N tren chan 4 va 2
  #if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(motorPin1, pwmFreq, pwmResolution);
    ledcAttach(motorPin2, pwmFreq, pwmResolution);
  #else
    ledcSetup(pwmChannel1, pwmFreq, pwmResolution);
    ledcSetup(pwmChannel2, pwmFreq, pwmResolution);
    ledcAttachPin(motorPin1, pwmChannel1);
    ledcAttachPin(motorPin2, pwmChannel2);
  #endif

  // Trang thai an toan ban dau: Dung dong co, tra thang servo
  setMotorSpeed(0);
  myServo.write(SERVO_CENTER);
  lastPacketTime = millis();
}

// -----------------------------------------------------------------------------
// 8. LOOP (VONG LAP CHINH)
// -----------------------------------------------------------------------------
void loop() {
  // 1. Doc va thuc thi goi tin dieu khien tu Raspberry Pi
  handleSerialInput();

  // 2. Watchdog Fail-Safe: Neu Pi mat ket noi qua 350ms -> Tu dong phanh dung xe
  if (!isStopped && (millis() - lastPacketTime > TIMEOUT_MS)) {
    setMotorSpeed(0);
    myServo.write(SERVO_CENTER);
    isStopped = true;
  }
}
