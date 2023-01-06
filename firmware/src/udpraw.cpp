#include "main.h"
#include <WiFiUdp.h>

#ifdef ENABLE_UDPRAW

#define INTERVAL_MS 1000

WiFiUDP udp;
TaskHandle_t udpTaskHandler;

void udpTask(void *pvParameters) {
    while (1)
	{

        camera_fb_t* fb = esp_camera_fb_get();

        udp.beginPacket("255.255.255.255", 8888);

        udp.write(fb->buf,fb->len);
        udp.endPacket();
        //Serial.println(String(fb->len)+"send");

        esp_camera_fb_return(fb);
        vTaskDelay(INTERVAL_MS/portTICK_PERIOD_MS);
    }
}

void initUDPraw(void){

    xTaskCreate(udpTask, "UDP", 8000, NULL, 1, &udpTaskHandler);



}

#endif