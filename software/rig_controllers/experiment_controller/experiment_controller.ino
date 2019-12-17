#include <Adafruit_NeoPixel.h>
#include <usb_serial.h>
#define SCB_AIRCR (*(volatile uint32_t *)0xE000ED0C) // Application Interrupt and Reset Control location

//define pins
const int lickPin = 35;
const int solenoidPin = 37;
const int ledPin = 34;
const int neoPin = 36;
const int lightsPin = 33;
const int robotOutPin = 38;
const int robotMovPin = 39;
const int robotRZPin = 14;
const int triggerPin = 15;

//robot
int robotOutState = 0;             //1 tells controller to move robot to next location
int robotRZState = 0;              //1 if robot has entered reward zone
int robotMovState = 0;             //0 if robot is in position, 1 if moving
int robotChanged = 0;              //1 if robot started moving (solves issues from robot not responding fast enough)

//reward delivery parameters
int flush_dur = 30000;              //time to keep solenoid open during initial water flush
int solenoid_open_dur = 75;          //time to keep solenoid open during single reward delivery
int solenoid_bounce_dur = 500;       //time between reward deliveries
int reward_win_dur = 3000;           //duration for which rewards could be delivered
int numRewards = 0;                //number of rewards that have been delivered in the current trial
int max_rewards = 3;                //maximum number of rewards that should be delivered in a trial
int solenoidOpen = 0;              //1 if reward is currently being delivered
int solenoidBounce = 0;            //1 if reward has been delivered within the bounce period
int inRewardWin = 0;               //1 if currently in the reward window
int lickState = 0;                 //1 if IR beam is broken

//lighting
int totalPixels = 384;
Adafruit_NeoPixel strip = Adafruit_NeoPixel(totalPixels, neoPin, NEO_RGBW + NEO_KHZ800);
uint32_t pureWhite = strip.Color(0,0,0,85); //don't go over 150!!!!!!
uint32_t pureOff = strip.Color(0,0,0,0);
int lights_off_dur = 3000;            //minimum time (ms) to keep lights off in between trials
int lights_on_dur = 5000;             //maximum time (ms) to keep tights on during a trial
int lightsOn = 0;                   //1 if lights are on
int ledOn = 0;                      //1 if led is on

//serial communication parameters
int baud_rate = 2000000;            //must be same as camera script
char serPNS = '0';                 //non-print-related serial commands from PNS
char newSerPNS = '0';              //instantaneous serial command from PNS
int handshake = 0;                 //1 if successful serial connection has been established
int expActive = 0;                 //1 when main experiment is being executed
int expEnded = 0;                  //1 if entire program should stop
String varName;                    //variable name for serial read/write
String protocol = "TRIALS";
int contMode = 0;

//timer variables
int waitDur = 5000;                //time (ms) to wait in between various events
int moveDelay = 500;               //time (ms) to wait after turning lights off before telling robot to move
int triggerPer = 5000;             //image trigger period in microseconds
int triggerDur = 1000;             //image trigger duration in microseconds
int triggerOn = 0;                 //1 if the trigger pin is HIGH
int triggered = 0;                 //1 if an image was triggered on the loop
int rwBuffDur = 500;               //time (ms) into reward window that images should continue being triggered
int reachDelay = 100;              //time (ms) after lights turn rats must wait before reaching
unsigned long solenoidInit;        //time since most recent solenoid opening
unsigned long lightsInit;          //time since lights turned on
unsigned long triggerInit = 0;     //time since most recent image trigger
unsigned long rewardWinInit = 0;   //time since reward window started
unsigned long reachInit = 0;       //time since reach was most recently detected

//counters
int rewardCount = 0; 
int trialCount = 0; 

void setup() {
  
  //initialize pins
  pinMode(lickPin,INPUT);
  pinMode(ledPin,OUTPUT);
  pinMode(solenoidPin,OUTPUT);
  pinMode(lightsPin,OUTPUT);
  pinMode(triggerPin,OUTPUT);
  pinMode(robotOutPin,OUTPUT);
  pinMode(robotMovPin,INPUT);
  pinMode(robotRZPin,INPUT);
  digitalWrite(ledPin,LOW);
  digitalWrite(solenoidPin,LOW);
  digitalWrite(lightsPin,LOW);
  digitalWrite(triggerPin,LOW);
  digitalWrite(robotOutPin,LOW);
  digitalWrite(robotMovPin,LOW);
  digitalWrite(robotRZPin,LOW);

  //start with lighting off
  strip.begin();
  strip.show();
  setAllPixels(totalPixels, pureOff);
  lightsOn = 0;

  //initiate serial connection
  Serial.begin(baud_rate);
  while (!Serial);
  while(handshake==0){
    if(Serial.available()){
      serPNS = Serial.read();
      if(serPNS == 'h'){
        Serial.print(serPNS);
        handshake = 1;
        Serial.flush();
      }
    }
  }

  //tell robot to move to start position
  digitalWrite(robotOutPin,HIGH);
  robotOutState = 1;
}

void loop() {

  //only execute when PNS has sent a command
  if (Serial.available() && expEnded==0) {
    
    //read command from PNS
    newSerPNS = Serial.read();
    switch(newSerPNS){
        case 'b':
          //begin experiment
          expActive = 1;
          serPNS = 's';
          break;
        case 'p':
          //pause experiment
          expActive = 0;
          break;
        case 'e':
          //end experiment
          expActive = 0;
          expEnded = 1;
          digitalWrite(robotOutPin,LOW);
          digitalWrite(solenoidPin,LOW);
          digitalWrite(ledPin,LOW);  
          setAllPixels(totalPixels, pureOff);
          digitalWrite(lightsPin,LOW);
          lightsOn = 0; 
          _softRestart();
          break;
        case 'n':
          //toggle neopixel lights       
          if(lightsOn==0){
            setAllPixels(totalPixels, pureWhite);
            digitalWrite(lightsPin,HIGH);
            lightsInit = millis();
            lightsOn = 1;
            serPNS = 'n';
          }else{
            setAllPixels(totalPixels, pureOff);
            digitalWrite(lightsPin,LOW);
            lightsInit = millis();
            lightsOn = 0;
            serPNS = 'n';
          }
          break;
        case 'r':
          //PNS detected a reach
          reachInit = millis();//start reach timer 
          serPNS = newSerPNS;
          if(!contMode && (reachInit-lightsInit)<reachDelay){
            //rat reached too soon, move to next trial
            inRewardWin = 1;
          }
          break;
        case 's':
          //PNS saved image buffer to disk   
          serPNS = newSerPNS;
          break;
        case 'w':
          //deliver water
          if(solenoidOpen==0 && solenoidBounce==0){
            digitalWrite(solenoidPin,HIGH);
            solenoidInit = millis();
            solenoidOpen = 1;
          }
          break;        
        case 'f':
          //flush water
          delay(waitDur);
          digitalWrite(solenoidPin,HIGH);
          delay(flush_dur);
          digitalWrite(solenoidPin,LOW);
          serPNS = '0';
          delay(waitDur);
          break;
        case 'l':
          //toggle led
          if(ledOn==0){
            digitalWrite(ledPin,HIGH);
            ledOn = 1;
          }else{
            digitalWrite(ledPin,LOW);
            ledOn = 0;
          }
          break;
        case 'm':
          //turn lights off and tell robot to move to next position
          if(robotOutState==0){
            lightsInit = millis(); 
            if(!contMode){           
              setAllPixels(totalPixels, pureOff);
              digitalWrite(lightsPin,LOW);
              lightsOn = 0;
              delay(moveDelay); //keeps lights off for some amount of time before telling the robot to move
            }               
            inRewardWin = 0;            
            digitalWrite(robotOutPin,HIGH);
            robotOutState = 1;
            robotChanged = 0;
            if(expActive==1){
              serPNS = 'e';
              trialCount++;
            }
          }
          break;
        case 't':
          //trigger an image
          if(!triggerOn){
            digitalWrite(triggerPin,HIGH);
            triggerInit = micros();
            triggerOn = 1;
          }
          break; 
        case 'g':
          //PNS requested to get a variable
          Serial.print(newSerPNS);
          while (!Serial.available()){}
          varName = Serial.readStringUntil('\n');
          if (varName=="flush_dur"){ 
            Serial.println(flush_dur);
          }else if(varName=="solenoid_open_dur"){
            Serial.println(solenoid_open_dur);
          }else if(varName=="solenoid_bounce_dur"){
            Serial.println(solenoid_bounce_dur);
          }else if(varName=="reward_win_dur"){
            Serial.println(reward_win_dur);
          }else if(varName=="max_rewards"){
            Serial.println(max_rewards);
          }else if(varName=="lights_off_dur"){
            Serial.println(lights_off_dur);
          }else if(varName=="lights_on_dur"){
            Serial.println(lights_on_dur);
          }else if(varName=="reachDelay"){
            Serial.println(reachDelay);
          }
          break;
        case 'v':
          //PNS requested to change a variable
          Serial.print(newSerPNS);
          while (!Serial.available()){}
          varName = Serial.readStringUntil('\n');
          Serial.print(newSerPNS);
          while (!Serial.available()){}
          if(varName=="flush_dur"){            
            flush_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="solenoid_open_dur"){
            solenoid_open_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="solenoid_bounce_dur"){
            solenoid_bounce_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="reward_win_dur"){
            reward_win_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="max_rewards"){
            max_rewards = Serial.readStringUntil('\n').toInt();
          }else if(varName=="lights_off_dur"){
            lights_off_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="lights_on_dur"){
            lights_on_dur = Serial.readStringUntil('\n').toInt();
          }else if(varName=="protocol"){
            protocol = Serial.readStringUntil('\n');
            if(protocol=="CONTINUOUS"){
              contMode = 1;
            }else if(protocol=="TRIALS"){
              contMode = 0;
            }
          }else if(varName=="reachDelay"){
            reachDelay = Serial.readStringUntil('\n').toInt();
          }
          break;                           
    }//end switch

    //execute main experiment?
    if (expActive==1){     

      //check robot
      robotMovState = digitalRead(robotMovPin);
      //should lighting change?
      if(lightsOn==0 && (millis()-lightsInit)>lights_off_dur && robotMovState==0 && serPNS=='s'){
        //lights are off, robot is in position, lights off timer is up, and image buffer has been saved
        //turn lights on and stop telling robot to move
        setAllPixels(totalPixels, pureWhite);
        digitalWrite(lightsPin,HIGH);
        lightsInit = millis();
        lightsOn = 1;
        digitalWrite(robotOutPin,LOW);
        robotOutState = 0;
      } else if(inRewardWin==1 && (millis()-rewardWinInit)>=reward_win_dur){
        //reward window just ended
        //turn lights off and tell robot to move to next position
        lightsInit = millis();
        if(!contMode){
          setAllPixels(totalPixels, pureOff);
          digitalWrite(lightsPin,LOW);
          lightsOn = 0; 
          delay(moveDelay); //keeps lights off for some amount of time before telling the robot to move
        }   
        inRewardWin = 0;
        digitalWrite(robotOutPin,HIGH);
        robotOutState = 1;
        robotChanged = 0;
        trialCount++;
        numRewards = 0;
        serPNS = 'e';
      } else if(contMode && robotMovState==0 && robotOutState==1 && (millis()-lightsInit)>lights_off_dur){
        //operating in CONTINUOUS MODE and lights off (e.g., intertrial interval) timer is up,
        //stop telling robot to move
        digitalWrite(robotOutPin,LOW);
        robotOutState = 0;
      }

      //trigger images
      if(!contMode && lightsOn && !triggerOn && !inRewardWin){
        digitalWrite(triggerPin,HIGH);
        triggerInit = micros();
        triggerOn = 1;
        triggered = 1;
      } else if(!contMode && lightsOn && !triggerOn && inRewardWin && (millis()-rewardWinInit)<=rwBuffDur){
        digitalWrite(triggerPin,HIGH);
        triggerInit = micros();
        triggerOn = 1;
        triggered = 1;
      } else if(contMode && lightsOn && !triggerOn) {
        digitalWrite(triggerPin,HIGH);
        triggerInit = micros();
        triggerOn = 1;
        triggered = 1;
      }

      //has robot entered reward zone?
      robotRZState = digitalRead(robotRZPin);
      if(!inRewardWin && robotOutState==0 && robotRZState==HIGH && serPNS == 'r'){
        rewardWinInit = millis();
        inRewardWin = 1;
      }
    
      //decide if solenoid should be opened
      lickState = digitalRead(lickPin);
      if(inRewardWin==1 && robotRZState==HIGH && lickState==HIGH && solenoidOpen==0 && solenoidBounce==0 && numRewards<max_rewards){   
        //reward window is active, robot is positioned in reward zone, lick is detected, and solenoid is ready
        //open solenoid
        digitalWrite(solenoidPin,HIGH);
        solenoidInit = millis();  
        solenoidOpen = 1;
        rewardCount++;
        numRewards++;
      }
      
      //echo PNS a line of data
      Serial.print(trialCount); 
      Serial.print(' ');   
      Serial.print(serPNS);  
      Serial.print(' ');     
      Serial.print(robotOutState); 
      Serial.print(' ');  
      Serial.print(triggered); 
      Serial.print(' ');  
      Serial.println(robotRZState); 
//      Serial.print(' ');  
//      Serial.println(robotRZState); 
//      Serial.print(' ');  
//      Serial.print(solenoidOpen); 
//      Serial.print(' ');  
//      Serial.println(lickState);
        triggered = 0;
    }else if(serPNS=='s') {
      //trigger images for baseline acquisition
      if(lightsOn && !triggerOn){
        serPNS = 'b';
        digitalWrite(triggerPin,HIGH);
        triggerInit = micros();
        triggerOn = 1;
      }
    }
  }
  //should trigger end?
  if(triggerOn && (micros()-triggerInit) >= triggerDur){
    digitalWrite(triggerPin, LOW);
    triggerOn = 0;
  }
  //should solenoid be closed?
  if(solenoidOpen==1 && (millis()-solenoidInit)>=solenoid_open_dur){
    digitalWrite(solenoidPin,LOW);
    solenoidInit = millis();
    solenoidOpen = 0;
    solenoidBounce = 1;    
  } else if(solenoidBounce==1 && (millis()-solenoidInit)>=solenoid_bounce_dur){
    solenoidBounce = 0;
  } 
  if(expActive==0){
    //should robot stop moving?
    if(robotOutState==1){
      robotMovState = digitalRead(robotMovPin);
      if(robotMovState==1){
        robotChanged = 1;
      }else if(robotMovState==0 && robotChanged==1){
        digitalWrite(robotOutPin,LOW);
        robotOutState = 0;
      }
    }
  }
}

void setAllPixels(int numPixel, uint32_t color){
  for(int currPixel = 0; currPixel < numPixel; currPixel++){
    strip.setPixelColor(currPixel, color);
  }
  strip.show();
}

void _softRestart() 
{
  Serial.end();  //clears the serial monitor  if used
  SCB_AIRCR = 0x05FA0004;  //write value for restart
}
