<p align="left">
  <img width="408" height="300" src="calibration_setup_img.png">
</p>

* For the Tactile Mode, you can use any vision-based tactile sensor (VBTS) technique, for instance, GelSight, GelSlim etc.
* For the Force Mode, you can also use any AR tag. Here we use ArUco marker and collected the groundtruth data from the ATI F/T sensor and the L3-FTOUCH sensor data using the setup shown above. Please make sure that the two data are synchronized and a multiple linear regression (MLR) model can be applied to obtain the calibration matrix of the sensor. The predicted data is presented in Linearity.mat, the overall output file is in XXX.py. 
