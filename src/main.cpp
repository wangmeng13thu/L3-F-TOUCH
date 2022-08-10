#include "main.h"
#include "wifikeys.h"

#define OTA

//config sensor role
#define ROLE DOUBLE_PALM_TOP

#define FINGER 0
#define THUMB_PALM 1
#define TRI_PALM 2
#define DOUBLE_PALM_BOT 3
#define DOUBLE_PALM_TOP 4

const int shutterTime[]={8,100,400,100,150};//finger,thumb_palm,tri_palm,double_palm
/** Camera class */
OV2640 cam;
boolean ota=false;

/** Function declarations */
void resetDevice(void);

camera_config_t userCamNew {

    .pin_pwdn = 19, //NC
    .pin_reset = -1,//NC

    .pin_xclk = 0,//MCLK

    .pin_sscb_sda = 8,//TWI SDA
    .pin_sscb_scl = 7,//TWI SCK

    // Note: LED GPIO is apparently 4 not sure where that goes
    // per https://github.com/donny681/ESP32_CAMERA_QR/blob/e4ef44549876457cd841f33a0892c82a71f35358/main/led.c
    .pin_d7 = 15,
    .pin_d6 = 13,
    .pin_d5 = 12,
    .pin_d4 = 27,
    .pin_d3 = 25,
    .pin_d2 = 32,
    .pin_d1 = 33,
    .pin_d0 = 26,
    .pin_vsync = 4,
    .pin_href = 2,//HSYNC
    .pin_pclk = 14,
    .xclk_freq_hz = 24000000,
    .ledc_timer = LEDC_TIMER_1,
    .ledc_channel = LEDC_CHANNEL_1,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
     //.frame_size = FRAMESIZE_QVGA, 
	 //.frame_size = FRAMESIZE_240X240,
#if ROLE
	.frame_size = FRAMESIZE_HQVGA,
#else
    .frame_size = FRAMESIZE_QCIF,
#endif
	//.frame_size = FRAMESIZE_QQVGA,
    .jpeg_quality = 10,               //0-63 lower numbers are higher quality
    .fb_count = 2 // if more than one i2s runs in continous mode.  Use only with jpeg
};

void setup()
{
	// Start the serial connection
	Serial.begin(115200);

	Serial.println("\n\n##################################");
	Serial.printf("Internal Total heap %d, internal Free Heap %d\n", ESP.getHeapSize(), ESP.getFreeHeap());
	Serial.printf("SPIRam Total heap %d, SPIRam Free Heap %d\n", ESP.getPsramSize(), ESP.getFreePsram());
	Serial.printf("ChipRevision %d, Cpu Freq %d, SDK Version %s\n", ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getSdkVersion());
	Serial.printf("Flash Size %d, Flash Speed %d\n", ESP.getFlashChipSize(), ESP.getFlashChipSpeed());
	Serial.println("##################################\n\n");

	
	delay(100);
	
	
	cam.init(userCamNew);
	delay(100);

	sensor_t * s = esp_camera_sensor_get();
  	s->set_wb_mode(s, 1);//enable manul WB as sunny
    s->set_exposure_ctrl(s,0);//enable manul shutter 
    s->set_saturation(s, 1);  
	s->set_aec_value(s,shutterTime[ROLE]);//shutter time

    //try adjust WB (seems RGB)
    s->set_reg(s,0XCC,0xFF,0x50);//R
    s->set_reg(s,0XCD,0xFF,0x41);//G
    s->set_reg(s,0XCE,0xFF,0x54);//B

	// Connect the WiFi
    Serial.println("Wifi init start");
	WiFi.mode(WIFI_STA);
	WiFi.begin(ssid, password);
	while (WiFi.status() != WL_CONNECTED)
	{
		delay(500);
		Serial.print(".");
	}


	// Print information how to contact the camera server
	IPAddress ip = WiFi.localIP();
	Serial.print("\nWiFi connected with IP ");
	Serial.println(ip);
#ifdef ENABLE_RTSPSERVER
	Serial.print("Stream Link: rtsp://");
	Serial.print(ip);
	Serial.println(":8554/mjpeg/1\n");
#endif
#ifdef ENABLE_WEBSERVER
	Serial.print("Browser Stream Link: http://");
	Serial.print(ip);
	Serial.println("\n");
	Serial.print("Browser Single Picture Link: http//");
	Serial.print(ip);
	Serial.println("/jpg\n");
#endif
#ifdef ENABLE_WEBSERVER
	// Initialize the HTTP web stream server
	initWebStream();
#endif

#ifdef ENABLE_RTSPSERVER
	// Initialize the RTSP stream server
	initRTSP();
#endif
	//use RX PIN for OTA 
	pinMode(3,INPUT_PULLUP);
}

void loop()
{	
#ifdef OTA

	if(digitalRead(3)==LOW){
		for(int i=0;i<10;i++){
			if (digitalRead(3)==HIGH) return;
			delay(1);
		}
		ota=true;
		stopRTSP();
		ArduinoOTA
		.onStart([]() {
		String type;
		if (ArduinoOTA.getCommand() == U_FLASH)
			type = "sketch";
		else // U_SPIFFS
			type = "filesystem";

		// NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
		Serial.println("Start updating " + type);
		})
		.onEnd([]() {
		Serial.println("\nEnd");
		})
		.onProgress([](unsigned int progress, unsigned int total) {
		Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
		})
		.onError([](ota_error_t error) {
		Serial.printf("Error[%u]: ", error);
		if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
		else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
		else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
		else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
		else if (error == OTA_END_ERROR) Serial.println("End Failed");
		});
		ArduinoOTA.begin();

	}
	if(ota)
		ArduinoOTA.handle();
#endif

	delay(100);
}


void resetDevice(void)
{
	delay(100);
	WiFi.disconnect();
	esp_restart();
}

