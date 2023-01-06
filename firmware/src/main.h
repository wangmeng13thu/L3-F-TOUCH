#include <Arduino.h>

// WiFi stuff
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiClient.h>

// OTA stuff
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

// Camera stuff
#include "OV2640.h"
#include "OV2640Streamer.h"
#include "CRtspSession.h"
#include "esp_camera.h"

//Select whether to use png in web and udp mode
//consuming ram and cpu
// #define USE_PNG 
void initEncoder(void);
extern int size_png;
extern uint8_t* buf_png;   // Memory to hold the output file
extern bool transing_png;

// Select which of the servers are active
// Select only one or the streaming will be very slow!
// #define ENABLE_WEBSERVER
#define ENABLE_RTSPSERVER
// #define ENABLE_UDPRAW

// Camera class
extern OV2640 cam;

// RTSP stuff
void initRTSP(void);
void stopRTSP(void);

// Web server stuff
void initWebStream(void);
void stopWebStream(void);
void handleWebServer(void);


//udpraw
void initUDPraw(void);