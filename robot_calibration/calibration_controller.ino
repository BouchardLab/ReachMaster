#include <stdint.h>

float durList[100] = {1.000000,  1.040307,  1.082238,  1.125859,  1.171238,  1.218447,  1.267558,  1.318649,  1.371799,  1.427091,  1.484613,  1.544452,  1.606704,
                       1.671464,  1.738835,  1.808921,  1.881833,  1.957683,  2.036590,  2.118678,  2.204074,  2.292913,  2.385332,  2.481477,  2.581496,  2.685547,
                       2.793792,  2.906400,  3.023547,  3.145416,  3.272197,  3.404087,  3.541294,  3.684031,  3.832522,  3.986997,  4.147699,  4.314879,  4.488796,
                       4.669724,  4.857944,  5.053751,  5.257450,  5.469359,  5.689810,  5.919147,  6.157727,  6.405923,  6.664123,  6.932731,  7.212165,  7.502862,
                       7.805276,  8.119880,  8.447164,  8.787639,  9.141838,  9.510314,  9.893641, 10.292419, 10.707271, 11.138844, 11.587811, 12.054876, 12.540765,
                       13.046240, 13.572088, 14.119131, 14.688224, 15.280255, 15.896149, 16.536867, 17.203410, 17.896820, 18.618178, 19.368612, 20.149292, 20.961440,
                       21.806322, 22.685259, 23.599622, 24.550841, 25.540399, 26.569843, 27.640781, 28.754884, 29.913893, 31.119617, 32.373940, 33.678820, 35.036296,
                       36.448486, 37.917597, 39.445923, 41.035850, 42.689862, 44.410541, 46.200575, 48.062758, 50.000000};
 
const int xPushPin = 23;
const int xPullPin = 22;
const int xPushVent = 21;
const int xPullVent = 20;
const int xPotPin = A0;
const int yPushPin = 19;
const int yPullPin = 18;
const int yPushVent = 17;
const int yPullVent = 16;
const int yPotPin = A1;
float xRandDur;
long xRandIdx;
long xRandPP;
float yRandDur;
long yRandIdx;
long yRandPP;
int offDur = 125;
char newSer = 'c';
int numOn = 15000;
bool xOn;
bool yOn;
unsigned long tOn;
float xPos;
float yPos;

void setup() {
  //start serial
  Serial.begin(38400);
  while (!Serial);
  
  //set pin modes
  pinMode(xPushPin, OUTPUT);
  pinMode(xPullPin, OUTPUT);
  pinMode(xPushVent, OUTPUT);
  pinMode(xPullVent, OUTPUT);
  pinMode(xPotPin, INPUT);
  pinMode(yPushPin, OUTPUT);
  pinMode(yPullPin, OUTPUT);
  pinMode(yPushVent, OUTPUT);
  pinMode(yPullVent, OUTPUT);
  pinMode(yPotPin, INPUT);

  //open vents
  digitalWrite(xPushVent, LOW);
  digitalWrite(xPullVent, LOW);
  digitalWrite(yPushVent, LOW);
  digitalWrite(yPullVent, LOW);
  delay(offDur);

  //random seed
  randomSeed(analogRead(A22));

  //wait for key to be pressed to start
  while(!Serial.available()){}
  
  //loop over onDur
  for(int i=0; i<numOn; i++){    
//    xRandPP = random(56,1024);    
//    yRandPP = random(109, 1024);
    xRandPP = random(1,3); 
    yRandPP = random(1, 3);
    xPos = analogRead(xPotPin);
    yPos = analogRead(yPotPin);

//    if(xPos<xRandPP && yPos<yRandPP){
    if(xRandPP==1 && yRandPP==1){
      //push both
      xRandIdx = random(0, 77);
      yRandIdx = random(0, 99);
      xRandDur = durList[xRandIdx]; 
      yRandDur = durList[yRandIdx];       
      digitalWrite(xPushPin,HIGH);
      digitalWrite(yPushPin,HIGH);
      tOn = micros();
      xOn = 1;
      yOn = 1;
      while(xOn || yOn){
        if(xOn && (micros()-tOn)>=xRandDur*1000){
          digitalWrite(xPushPin,LOW);
          xOn = 0;
        }
        if(yOn && (micros()-tOn)>=yRandDur*1000){
          digitalWrite(yPushPin,LOW);
          yOn = 0;
        }
      }
      delay(offDur);
    }
//    else if(xPos<xRandPP && yPos>yRandPP){     
    else if(xRandPP==1 && yRandPP==2){   
      //push x pull y
      xRandIdx = random(0, 77);
      yRandIdx = random(0, 99);
      xRandDur = durList[xRandIdx]; 
      yRandDur = durList[yRandIdx]; 
      digitalWrite(xPushPin,HIGH);
      digitalWrite(yPullPin,HIGH);
      tOn = micros();
      xOn = 1;
      yOn = 1;
      while(xOn || yOn){
        if(xOn && (micros()-tOn)>=xRandDur*1000){
          digitalWrite(xPushPin,LOW);
          xOn = 0;
        }
        if(yOn && (micros()-tOn)>=yRandDur*1000){
          digitalWrite(yPullPin,LOW);
          yOn = 0;
        }
      }
      delay(offDur);
      
    }
//    else if(xPos>xRandPP && yPos<yRandPP){
    else if(xRandPP==2 && yRandPP==1){
      //pull x push y
      xRandIdx = random(0, 99);
      yRandIdx = random(0, 99);
      xRandDur = durList[xRandIdx]; 
      yRandDur = durList[yRandIdx]; 
      digitalWrite(xPullPin,HIGH);
      digitalWrite(yPushPin,HIGH);
      tOn = micros();
      xOn = 1;
      yOn = 1;
      while(xOn || yOn){
        if(xOn && (micros()-tOn)>=xRandDur*1000){
          digitalWrite(xPullPin,LOW);
          xOn = 0;
        }
        if(yOn && (micros()-tOn)>=yRandDur*1000){
          digitalWrite(yPushPin,LOW);
          yOn = 0;
        }
      }
      delay(offDur);
      
    }
    else {
      //pull both
      xRandIdx = random(0, 99);
      yRandIdx = random(0, 99);
      xRandDur = durList[xRandIdx]; 
      yRandDur = durList[yRandIdx]; 
      digitalWrite(xPullPin,HIGH);
      digitalWrite(yPullPin,HIGH);
      tOn = micros();
      xOn = 1;
      yOn = 1;
      while(xOn || yOn){
        if(xOn && (micros()-tOn)>=xRandDur*1000){
          digitalWrite(xPullPin,LOW);
          xOn = 0;
        }
        if(yOn && (micros()-tOn)>=yRandDur*1000){
          digitalWrite(yPullPin,LOW);
          yOn = 0;
        }
      }
      delay(offDur);
    }    
  }
}

void loop() {
  
}
