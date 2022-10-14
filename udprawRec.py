from socket import * 
from time import *
from PIL import Image

HOST = ''
PORT = 8888
BUFSIZ = 38400
ADDRESS = (HOST, PORT)
W = 160
H = 120
IMSIZ = W*H*2

udpServerSocket = socket(AF_INET, SOCK_DGRAM)
udpServerSocket.bind(ADDRESS)      # 绑定客户端口和地址

while True:
    print("waiting for message...")

    imData=bytes()
    timer=0
    pac=0

    while len(imData)<IMSIZ:
        
        data, addr = udpServerSocket.recvfrom(BUFSIZ)
        if (time()-timer>0.5):
            imData=bytes()
            pac=0
        imData+=data
        timer=time()
        pac+=1
        print(pac)
        print(len(imData))
            


    rgb=bytes()
    for p in range(0,IMSIZ,2):
        v=(imData[p]<<8)+imData[p+1]
        b = v&0b11111
        g = (v>>5)&0b111111
        r = (v>>11)&0b11111
        b = int(b*255/0b11111)
        g = int(g*255/0b111111)
        r = int(r*255/0b11111)
        rgb+=bytes([r,g,b])

    image = Image.frombytes('RGB', (W,H), rgb)
    image.show()
