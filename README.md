L3 F-TOUCH Design
=================

<p align="left">
  <img width="272" height="200" src="prototype/sensor_whole.png">
</p>

L3 F-TOUCH sensor is an enhanced version of the GelSight sensor, it acquires a much better three-axis force sensing capability while being ***L***ight-weight, ***L***ow-cost and wire***L***ess for the ease of replication and deployment.

This repository contains the documentation and manufacturing files for L3 F-TOUCH tactile sensor.

Manufacturing
-------------

* The 3D design files and BOM are in [prototype](prototype/);
* The wireless camera module design is in [PCB](PCB/);

Running
-----------

* Compile and upload the [firmware](/firmware) via [PlatformIO](https://github.com/platformio/platformio-vscode-ide) (WiFi must be configured);
* Use [VLC](https://github.com/videolan/vlc) to check the RTSP video streaming. Adress can be found in Serial monitor when startup.
* Run [L3-FTOUCH.py](/software/L3-FTOUCH.py) for demonstration;

Usage
-----

* Run [xx.py]() to get data from streaming (default).
* Modify the firmware to use picture mode for higher quality images (advanced).
