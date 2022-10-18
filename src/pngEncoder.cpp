#include "main.h"
#include <PNGenc.h>

#define PNG_BUF 600000
#define COMPRESS_LEVEL 1 // ZLIB compression levels: 1 = fastest, 9 = most compressed (slowest)


PNG png; // static instance of the PNG encoder lass
int size_png;
TaskHandle_t pngTaskHandler;
// Memory to hold the output file
uint8_t* outA;
uint8_t* outB;

uint8_t *buf_png = outA;


void pngTask(void *pvParameters)
{
    while (1)
    {
        camera_fb_t *fb = esp_camera_fb_get();
        uint8_t *buf = fb->buf;
        int WIDTH = fb->width;
        int HEIGHT = fb->height;

        int time_ms = millis();
        uint8_t* writeBuf = buf_png==outA?outB:outA;

        
        int rc = png.open(writeBuf, PNG_BUF);
        if (rc == PNG_SUCCESS)
        {

            rc = png.encodeBegin(WIDTH, HEIGHT, PNG_PIXEL_TRUECOLOR, 3*8, NULL, COMPRESS_LEVEL); //Bpp: bits per pixel

            if (rc == PNG_SUCCESS)
            {
                for (int y = 0; y < HEIGHT && rc == PNG_SUCCESS; y++)
                {
                    uint8_t tempLine[WIDTH * 3];
                          
                    for (int x = 0; x < WIDTH; x++){
                        uint16_t v = (buf[2*x]<<8)+buf[2*x+1];
                        uint8_t b = v&0b11111;
                        uint8_t g = (v>>5)&0b111111;
                        uint8_t r = (v>>11)&0b11111;
                        tempLine[x*3+2] = int(b*255/0b11111);
                        tempLine[x*3+1] = int(g*255/0b111111);
                        tempLine[x*3] = int(r*255/0b11111);
                    }

                    rc = png.addLine(tempLine);
                    //Serial.println(rc);
                    buf += WIDTH * 2;
                } // for y
                size_png = png.close();
                buf_png = writeBuf;


                Serial.printf("%d bytes of data written to file in %d ms\n", size_png, millis() - time_ms);
            }
        }

        esp_camera_fb_return(fb);

        vTaskDelay(1);
    }
}

void initEncoder()
{
    outA = (uint8_t* )ps_malloc(PNG_BUF);
    outB = (uint8_t* )ps_malloc(PNG_BUF);
    xTaskCreatePinnedToCore(pngTask, "PNG", 10000, NULL, tskIDLE_PRIORITY + 1, &pngTaskHandler,0);
}