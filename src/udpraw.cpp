#include "main.h"
#include <WiFiUdp.h>

#ifdef ENABLE_UDPRAW

WiFiUDP udp;



void initUDPraw(void){
    camera_fb_t* fb = esp_camera_fb_get();

    udp.beginPacket("255.255.255.255", 8888);

    udp.write(fb->buf,fb->len);
    udp.endPacket();
    Serial.println(String(fb->len)+"send");

    esp_camera_fb_return(fb);

}

#endif