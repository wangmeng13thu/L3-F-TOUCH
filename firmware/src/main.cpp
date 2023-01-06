#include "main.h"
#include "wifikeys.h"

#define OTA

//config sensor role
#define ROLE FINGER

#define FINGER 0
#define THUMB_PALM 1
#define TRI_PALM 2
#define DOUBLE_PALM_BOT 3
#define DOUBLE_PALM_TOP 4

const int shutterTime[]={1,110,50,20,20};//finger default is 8,thumb_palm,tri_palm 350,double_palm
/** Camera class */
OV2640 cam;
boolean ota=false;

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
    .xclk_freq_hz = 24000000, //24000000
    .ledc_timer = LEDC_TIMER_1,
    .ledc_channel = LEDC_CHANNEL_1,


#ifdef USE_PNG
	.pixel_format = PIXFORMAT_RGB565,	
#if ROLE
	.frame_size = FRAMESIZE_VGA,
#else
    .frame_size = FRAMESIZE_VGA,
#endif

#else
	 .pixel_format = PIXFORMAT_JPEG,

#ifdef ENABLE_RTSPSERVER
#if ROLE
	.frame_size = FRAMESIZE_HQVGA,	
#else
    .frame_size = FRAMESIZE_QCIF,
#endif

#else // ENABLE_WEBSERVER
#if ROLE
	.frame_size = FRAMESIZE_HQVGA,
#else
    .frame_size = FRAMESIZE_QCIF,
#endif

#endif
#endif


#ifdef ENABLE_WEBSERVER	
    .jpeg_quality = 3,             //0-63 lower numbers are higher quality	10
    .fb_count = 1 // if more than one i2s runs in continous mode.  Use only with jpeg
#else //enable rtspserver
	.jpeg_quality = 10,             //0-63 lower numbers are higher quality	10
    .fb_count = 2 // if more than one i2s runs in continous mode.  Use only with jpeg
#endif

};

camera_config_t test {

    .pin_pwdn = 32,
    .pin_reset = -1,

    .pin_xclk = 0,

    .pin_sscb_sda = 26,
    .pin_sscb_scl = 27,

    // Note: LED GPIO is apparently 4 not sure where that goes
    // per https://github.com/donny681/ESP32_CAMERA_QR/blob/e4ef44549876457cd841f33a0892c82a71f35358/main/led.c
    .pin_d7 = 35,
    .pin_d6 = 34,
    .pin_d5 = 39,
    .pin_d4 = 36,
    .pin_d3 = 21,
    .pin_d2 = 19,
    .pin_d1 = 18,
    .pin_d0 = 5,
    .pin_vsync = 25,
    .pin_href = 23,
    .pin_pclk = 22,
    .xclk_freq_hz = 24000000,
    .ledc_timer = LEDC_TIMER_1,
    .ledc_channel = LEDC_CHANNEL_1,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
    // .frame_size = FRAMESIZE_VGA, // needs 96K or even smaller FRAMESIZE_SVGA - can work if using only 1 fb
    // .frame_size = FRAMESIZE_QCIF,
 .frame_size = FRAMESIZE_VGA,
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
  	// s->set_wb_mode(s, 1);//enable manul WB as sunny
    // s->set_exposure_ctrl(s,0);//enable manul shutter 
    // s->set_saturation(s, 1);  
	// // s->set_sharpness(s, 2);
	// s->set_aec_value(s,shutterTime[ROLE]);//shutter time

    //try adjust WB (seems RGB)
    // s->set_reg(s,0XCC,0xFF,0x52);//R default:0x50 3.50:0x48 3.53:0x48 3.61:0x52 3.60:0x55 3.59:0x48 3.56:0x54 3.68:0x50 3.55:0x53 3.58:0x54 3.63:0x50 3.52:0x54 3.54:0x52 3.62  3.66:0x50 3.51:0x48 3.53:0x52 3.57:0x52 3.61 0x48 3.65:0x48
    // s->set_reg(s,0XCD,0xFF,0x45);//G default:0x41 3.50:0x45 3.53:0x41 3.61:0x45 3.60:0x45 3.59:0x45 3.56:0x45 3.68:0x41 3.55:0x42 3.58:0x44 3.63:0x48 3.52:0x45 3.54:0x41 3.62  3.66:0x45 3.51:0x41 3.53:0x41 3.57:0x45 3.61 0x45 3.65:0x41
    // s->set_reg(s,0XCE,0xFF,0x50);//B default:0x54 3.50:0x52 3.53:0x50 3.61:0x50 3.60:0x54 3.59:0x52 3.56:0x54 3.68:0x42 3.55:0x54 3.58:0x54 3.63:0x52 3.52:0x54 3.54:0x52 3.62  3.66:0x52 3.51:0x50 3.53:0x52 3.57:0x48 3.61 0x52 3.65:0x50

	//top finger default RGB : 0x52 0x45 0x50
	//middle finger default RGB : 0x50 0x42 0x52
	//bottom finger default RGB : 0x46 0x42 0x48
	//TRI_PALM default: 0x46 0x42 0x48

	// Connect the WiFi
    Serial.println("Wifi init start");
	WiFi.mode(WIFI_STA);
	WiFi.begin(ssid, password);
	//WiFi.setTxPower(WIFI_POWER_8_5dBm);
	Serial.println("TX Power: "+String(WiFi.getTxPower()));

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
	// Initialize the RTSP stream server
	initRTSP();
#endif
#ifdef ENABLE_WEBSERVER
	Serial.print("Browser Stream Link: http://");
	Serial.print(ip);
	Serial.println("\n");
	Serial.print("Browser Single Picture Link: http://");
	Serial.print(ip);
	Serial.println("/jpg\n");
	// Initialize the HTTP web stream server
	initWebStream();
#endif
#ifdef ENABLE_UDPRAW
	Serial.println("Broadcast via UDP, to port 8888");
	initUDPraw();
#endif
#ifdef USE_PNG
	initEncoder();
	s->set_reg(s,0XDA,0x01,0x01); //change high byte first/low byte first of RGB565 data
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
		#ifdef ENABLE_RTSPSERVER
		stopRTSP();
		#endif
		#ifdef ENABLE_WEBSERVER
		stopWebStream();
		#endif

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
		ArduinoOTA.setTimeout(5000);

	}
	if(ota)
		ArduinoOTA.handle();
#endif

	delay(10);
}
