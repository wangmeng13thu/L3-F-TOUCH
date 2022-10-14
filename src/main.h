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


// Select which of the servers are active
// Select only one or the streaming will be very slow!
//#define ENABLE_WEBSERVER
//#define ENABLE_RTSPSERVER
#define ENABLE_UDPRAW

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