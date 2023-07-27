import numpy as np
import cv2
import sys
# from utils import ARUCO_DICT
import argparse
import time
import math
import matplotlib.pyplot as plt
import threading
import nidaqmx
import tkinter as tk
import csv
from PIL import Image, ImageTk
import imutils
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sys.path.append('TestWifiCamera/GelForce')
import setting_GelForce
from lib import find_marker
from collections import deque
from fast_poisson import fast_poisson
from fast_poisson1 import poisson_reconstruct



class Processing:
    def __init__(self, vs, aruco_dict_type):
        self.vs = vs
        self.aruco_dict_type = aruco_dict_type
        self.frame = None
        self.frame_Gel = None
        self.thread1 = None
        self.thread2 = None
        self.stopEvent = None
        self.writedata = False
        self.resetsensor = False
        self.counter = 0
        self.scale = 1

        self.ab_array = np.load('./correction.npz')
        self.x_index = self.ab_array['x']
        self.y_index = self.ab_array['y']
        self.FT_Cali = np.zeros((6,6))

        self.cameraMatrix = np.float32([[160.1881, 0, 247.7054],
                                        [0, 160.3019, 170.0790],
                                        [0, 0, 1.0]])

        self.Sensor_Cali = np.load('./forcematrix.npy')
        # self.Sensor_Cali = np.float32([[625.5201, 110.8314, -532.1205],
        #                                [203.4257, -993.7571, -671.1609],
        #                                [-1492.8, 112.8753, 5734.2]])

        self.disCoeffs = np.float32([0.2616 , -0.2155, 0.0032, 0.0022, 0.0226])

        self.ATIdata = np.zeros((6,1))
        self.FT = {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0, 'Tx': 0.0, 'Ty': 0.0, 'Tz': 0.0}
        self.ati_bias = np.zeros((6, 1))
        self.sensor_bias = np.zeros((3, 1))
        self.tvec = np.zeros((3, 1))
        self.pre_tvec = np.zeros((3, 1))
        self.tvecFT = np.zeros((3, 1))
        self.idcheck = True


        self.FrameGel_inix = 185
        self.FrameGel_iniy = 35
        self.FrameGel_width = 120
        self.FrameGel_height = 160
        self.con_flag1 = True
        self.reset_shape1 = True
        self.restart1 = False
        self.refresh1 = False
        self.writeTacdata = False
        self.slip_indicator1 = False
        self.slip_indicator2 = False
        # abe_array = np.load('abe_corr.npz')
        # self.x_index = abe_array['x']
        # self.y_index = abe_array['y']

        self.collideThre = 3    # 2.5
        self.collide_rotation = 4.
        self.showimage1 = True
        self.showimage2 = False
        self.data1 = deque(maxlen=75)
        self.collision_detected = False
        self.slip = True
        self.rotateslip = True
        self.calibrate1 = False
        self.taccount = 110

        self.table = np.load('./table_smooth.npy')
        self.zeropts_scale_bin = np.load('./zeropts_scale_bin.npy')
        # self.zeropoint =  [-55, -30, -55]
        # self.lookscale = [120., 60., 120.]
        # self.bin_num = [60, 60, 60]
        self.zeropoint = self.zeropts_scale_bin[0]
        self.lookscale = self.zeropts_scale_bin[1]
        self.bin_num = self.zeropts_scale_bin[2]

        self.useTable = True
        self.useMLP = False
        self.useMask = True

        self.root = tk.Tk()
        canvas = tk.Canvas(self.root, height=640, width=960, bg='gray')
        canvas.pack()
        self.panel1 = None
        self.panel2 = None
        self.panel3 = None
        self.label1 = None
        self.label2 = None

        button1 = tk.Button(canvas, height=2, width=6, text="Start", font=("Times", 20), bg='white', command=self.startsensor)
        button1.place(relx=0.05, rely=0.01, relwidth=0.25, relheight=0.05)
        button2 = tk.Button(canvas, height=2, width=6, text="Record", font=("Times", 20), bg='white', command=self.plotdata)
        button2.place(relx=0.375, rely=0.01, relwidth=0.25, relheight=0.05)
        button3 = tk.Button(canvas, height=2, width=6, text="Quit", font=("Times", 20), bg='white', command=self.quitsensor)
        button3.place(relx=0.7, rely=0.01, relwidth=0.25, relheight=0.05)

        self.panel1 = tk.Label(canvas)
        self.panel1.place(relx=0.05, rely=0.1, relwidth=0.2, relheight=0.3)    #0.08 0.3 0.5
        self.panel2 = tk.Label(canvas)
        self.panel2.place(relx=0.35, rely=0.1, relwidth=0.2, relheight=0.3)    #0.08 0.3 0.5
        self.panel3 = tk.Label(canvas)
        self.panel3.place(relx=0.65, rely=0.1, relwidth=0.2, relheight=0.3)  # 0.08 0.3 0.5

        self.label1 = tk.Label(canvas, height=2, width=18, bg='white')
        self.label1.place(relx=0.05, rely=0.55, relwidth=0.65, relheight=0.3)
        self.label2 = tk.Label(canvas, height=2, width=18, bg='white', font=("Arial", 12))
        self.label2.place(relx=0.75, rely=0.55, relwidth=0.2, relheight=0.3)
        #self.task = None


        self.stopEvent = threading.Event()
        self.thread1 = threading.Thread(target=self.videoloop, args=())
        print("start threading1")
        self.thread1.start()

        # self.thread2 = threading.Thread(target=self.atiloop, args=())
        # print("start threading2")
        # self.thread2.start()
        #
        # self.root.mainloop()

        self.root.wm_title("Force/Tactile Sensor")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onclose)

    def atiloop(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("Dev2/ai0:5")
            while True:
                self.ATIdata = task.read(number_of_samples_per_channel=1)
                self.ATIdata = np.array(self.ATIdata)


    def pose_esitmation(self, frame, aruco_dict_type):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters, cameraMatrix=self.cameraMatrix,
                                                                    distCoeff=self.disCoeffs)
        # print(f"ids are {ids}")
        # If markers are detected
        if len(corners) > 0 and ids[0] == [0] and len(ids) == 1:
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.003, self.cameraMatrix, self.disCoeffs)
                # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis
            cv2.aruco.drawAxis(frame, self.cameraMatrix, self.disCoeffs, rvec, tvec, 0.002)

            self.tvec = tvec[:, :, [1, 0, 2]]
            self.tvec = self.tvec.reshape(3, 1)
            self.tvec[1,:] *= -1
            self.pre_tvec = self.tvec.copy()

        if ids is None:
            self.idcheck = False
        else:
            self.idcheck = True
                # print(f"tvec OG is {tvec} and OG shape is {tvec.shape}")
                # # print(f"tvec1 is {tvec1} and 1 shape is {tvec1.shape}")
                # print(f"tvec self is {self.tvec} and self shape is {self.tvec.shape}")

        return frame

    def make_kernel(self, n, k_type):
        if k_type == 'circle':
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernal

    def marker_detection(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        # raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        raw_image_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (3, 3), 0).astype(np.uint8)
        ref_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (25, 25), 0).astype(np.uint8)

        diff = cv2.subtract(ref_blur, raw_image_blur)
        # diff = raw_image_blur - ref_blur
        diff *= 16

        diff[diff < 0] = 0
        diff[diff > 255] = 255

        mask = ((diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) &
                (diff[:, :, 1] > 120)).astype(np.uint8)
        mask *= 255
        # mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel2, iterations=1)
        mask = cv2.erode(mask, self.kernel2, iterations=1)
        mask = cv2.bitwise_not(mask)

        return mask

    def creat_mask_2(self, raw_image, dmask):
        # t = time.time()
        scale = 2
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (15, 15), 0)
        blur2 = cv2.GaussianBlur(raw_image, (3, 3), 0)
        # print(time.time() - t)
        diff = blur - blur2
        # diff = cv2.resize(diff, (int(m / scale), int(n / scale)))
        # diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255
        diff *= 15.0
        # cv2.imshow('blur2', blur.astype(np.uint8))
        # cv2.waitKey(1)

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        # diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(1)
        mask_b = diff[:, :, 0] > 150
        mask_g = diff[:, :, 1] > 150
        mask_r = diff[:, :, 2] > 150

        mask = ((mask_b * mask_g) + (mask_b * mask_r) + (mask_g * mask_r)) > 0
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(1)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = mask * dmask
        kernal4 = self.make_kernel(3, 'circle')
        mask = cv2.dilate(mask, kernal4, iterations=1)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(1)
        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        # print(time.time() - t)
        return (1 - mask) * 255

    def find_dots(self, binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.thresholdStep = 5    #0
        params.minThreshold = 1     # 1
        params.maxThreshold = 120    # 40
        params.minDistBetweenBlobs = 9  #9
        params.filterByArea = True
        params.minArea = 12  # 9
        params.maxArea = 1000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5 #0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))

        return keypoints

    def make_mask(self, img, keypoints):
        img = np.zeros_like(img[:, :, 0])
        for i in range(len(keypoints)):
            # cv2.circle(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (1), -1)
            cv2.ellipse(img,
                        (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])),
                        (8, 6), 0, 0, 360, (255), -1)  #original 10,8
        # cv2.imshow("makemask", img)
        return img

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.33, 0.33, 0.34])

    def defect_mask(self, im_cal):
        # mask = np.load('./TacData_fill/defect_mask.npy')
        # pad, var0, var1, var2, var3 = int(mask[0]), int(mask[1]), int(mask[2]), int(mask[3]), int(mask[4])
        pad = 1
        var0 = 15  # left up 60
        var1 = 15  # right up 60
        var2 = 15  # right down 65
        var3 = 15  # left down 60
        im_mask = np.ones((im_cal.shape))
        triangle0 = np.array([[0, 0], [var0, 0], [0, var0]])
        triangle1 = np.array([[im_mask.shape[1] - var1, 0],
                              [im_mask.shape[1], 0], [im_mask.shape[1], var1]])
        triangle2 = np.array([[im_mask.shape[1] - var2, im_mask.shape[0]], [im_mask.shape[1], im_mask.shape[0]],
                              [im_mask.shape[1], im_mask.shape[0] - var2]])
        triangle3 = np.array([[0, im_mask.shape[0]],
                              [0, im_mask.shape[0] - var3],
                              [var3, im_mask.shape[0]]])
        color = [0]  # im_mask
        cv2.fillConvexPoly(im_mask, triangle0, color)
        cv2.fillConvexPoly(im_mask, triangle1, color)
        cv2.fillConvexPoly(im_mask, triangle2, color)
        cv2.fillConvexPoly(im_mask, triangle3, color)
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad] = 0
        im_mask[:, -pad:] = 0
        return im_mask

    def defect_mask_diff(self, im_cal):
        mask = np.load('./TacData_fill/defect_mask.npy')
        pad, var0, var1, var2, var3 = int(mask[0]), int(mask[1]), int(mask[2]), int(mask[3]), int(mask[4])
        im_mask = im_cal.copy()
        triangle0 = np.array([[0, 0], [var0, 0], [0, var0]])
        triangle1 = np.array([[im_mask.shape[1] - var1, 0],
                              [im_mask.shape[1], 0], [im_mask.shape[1], var1]])
        triangle2 = np.array([[im_mask.shape[1] - var2, im_mask.shape[0]], [im_mask.shape[1], im_mask.shape[0]],
                              [im_mask.shape[1], im_mask.shape[0] - var2]])
        triangle3 = np.array([[0, im_mask.shape[0]],
                              [0, im_mask.shape[0] - var3],
                              [var3, im_mask.shape[0]]])
        color = [0]  # im_mask
        cv2.fillConvexPoly(im_mask, triangle0, color)
        cv2.fillConvexPoly(im_mask, triangle1, color)
        cv2.fillConvexPoly(im_mask, triangle2, color)
        cv2.fillConvexPoly(im_mask, triangle3, color)
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad] = 0
        im_mask[:, -pad:] = 0
        return im_mask

    def defect_mask0(img):
        im_mask = np.ones((img.shape))

        return im_mask.astype(int)

    def sortkeypoints(self, keypoints):
        x, y, xy = [], [], []
        # print(f"keypoint size is {len(keypoints)}")
        for i in range(len(keypoints)):
            x.append(keypoints[i].pt[0])
            y.append(keypoints[i].pt[1])
            xy.append((int(keypoints[i].pt[0]), int(keypoints[i].pt[1])))
        # xy = sorted(xy)
        # xy = sorted(xy, key=lambda x : x[1])
        xy = sorted(xy, key=lambda x: [x[1], x[0]])
        # xy_array = []
        # temp = []
        # for i in range(len(xy)):
        #     xy_array.append(i)
        # print("xy_array is", xy)
        # print("len of xy_array is", len(xy_array))
        return xy

    def draw_flow(self, frame, flow):
        Ox, Oy, Cx, Cy, Occupied = flow
        K = 0
        for i in range(len(Ox)):
            for j in range(len(Ox[i])):
                pt1 = (int(Ox[i][j]), int(Oy[i][j]))
                pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
                color = (255, 255, 0)
                if Occupied[i][j] <= -1:
                    color = (127, 127, 255)
                cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

    def flow_calculate_global_1(self, keypoints2):

        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []
        x1_return, y1_return, x2_return, y2_return, u_return, v_return = [], [], [], [], [], []

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i,0] / self.scale)
            y2.append(keypoints2[i,1] / self.scale)
            # x2.append(keypoints2[i].pt[0] / self.scale)
            # y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)

        for i in range(x2.shape[0]):
            distance = list(((np.array(self.x_iniref1) - x2[i]) ** 2 +
                             (np.array(self.y_iniref1) - y2[i]) ** 2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - self.x_iniref1[min_index]
            v_temp = y2[i] - self.y_iniref1[min_index]
            shift_length = np.sqrt(u_temp ** 2 + v_temp ** 2)
            # print 'length',shift_length
            if shift_length < 12:
                x1_paired.append(self.x_iniref1[min_index] -
                                 self.u_addon1[min_index])
                y1_paired.append(self.y_iniref1[min_index] -
                                 self.v_addon1[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + self.u_addon1[min_index])
                v.append(v_temp + self.v_addon1[min_index])

                del self.x_iniref1[min_index], self.y_iniref1[
                    min_index], self.u_addon1[min_index], self.v_addon1[
                    min_index]

        x1_return = np.array(x1_paired)
        y1_return = np.array(y1_paired)
        x2_return = np.array(x2_paired)
        y2_return = np.array(y2_paired)
        u_return = np.array(u)
        v_return = np.array(v)
        self.x_iniref1 = list(x2_paired)
        self.y_iniref1 = list(y2_paired)
        self.u_addon1 = list(u)
        self.v_addon1 = list(v)

        return x1_return, y1_return, x2_return, y2_return, u_return, v_return

    def flow_calculate_in_contact_1(self, keypoints2):

        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []
        temp_x = list(self.x_iniref1)
        temp_y = list(self.y_iniref1)
        temp_u = list(self.u_addon1)
        temp_v = list(self.v_addon1)

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i,0] / self.scale)
            y2.append(keypoints2[i,1] / self.scale)
            # x2.append(keypoints2[i].pt[0] / self.scale)
            # y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)
        refresh = False

        for i in range(x2.shape[0]):

            distance = list(((np.array(temp_x) - x2[i]) ** 2 +
                             (np.array(temp_y) - y2[i]) ** 2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - temp_x[min_index]
            v_temp = y2[i] - temp_y[min_index]
            shift_length = np.sqrt(u_temp ** 2 + v_temp ** 2)
            # print 'length',shift_length

            if shift_length < 12:   #12
                # print xy2.shape,min_index,len(distance)
                x1_paired.append(temp_x[min_index] - temp_u[min_index])
                y1_paired.append(temp_y[min_index] - temp_v[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + temp_u[min_index])
                v.append(v_temp + temp_v[min_index])

                del temp_x[min_index], temp_y[min_index], temp_u[
                    min_index], temp_v[min_index]

                if shift_length > 10:   #10
                    refresh = True

        return x1_paired, y1_paired, x2_paired, y2_paired, u, v, refresh

    def flow_calculate_global_2(self, keypoints2):

        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []
        x1_return, y1_return, x2_return, y2_return, u_return, v_return = [], [], [], [], [], []

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i,0] / self.scale)
            y2.append(keypoints2[i,1] / self.scale)
            # x2.append(keypoints2[i].pt[0] / self.scale)
            # y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)

        for i in range(x2.shape[0]):
            distance = list(((np.array(self.x_iniref2) - x2[i]) ** 2 +
                             (np.array(self.y_iniref2) - y2[i]) ** 2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - self.x_iniref2[min_index]
            v_temp = y2[i] - self.y_iniref2[min_index]
            shift_length = np.sqrt(u_temp ** 2 + v_temp ** 2)
            # print 'length',shift_length
            if shift_length < 12:
                x1_paired.append(self.x_iniref2[min_index] -
                                 self.u_addon2[min_index])
                y1_paired.append(self.y_iniref2[min_index] -
                                 self.v_addon2[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + self.u_addon2[min_index])
                v.append(v_temp + self.v_addon2[min_index])

                del self.x_iniref2[min_index], self.y_iniref2[
                    min_index], self.u_addon2[min_index], self.v_addon2[
                    min_index]

        x1_return = np.array(x1_paired)
        y1_return = np.array(y1_paired)
        x2_return = np.array(x2_paired)
        y2_return = np.array(y2_paired)
        u_return = np.array(u)
        v_return = np.array(v)
        self.x_iniref2 = list(x2_paired)
        self.y_iniref2 = list(y2_paired)
        self.u_addon2 = list(u)
        self.v_addon2 = list(v)

        return x1_return, y1_return, x2_return, y2_return, u_return, v_return

    def flow_calculate_in_contact_2(self, keypoints2):

        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []
        temp_x = list(self.x_iniref2)
        temp_y = list(self.y_iniref2)
        temp_u = list(self.u_addon2)
        temp_v = list(self.v_addon2)

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i,0] / self.scale)
            y2.append(keypoints2[i,1] / self.scale)
            # x2.append(keypoints2[i].pt[0] / self.scale)
            # y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)
        refresh = False

        for i in range(x2.shape[0]):

            distance = list(((np.array(temp_x) - x2[i]) ** 2 +
                             (np.array(temp_y) - y2[i]) ** 2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - temp_x[min_index]
            v_temp = y2[i] - temp_y[min_index]
            shift_length = np.sqrt(u_temp ** 2 + v_temp ** 2)
            # print 'length',shift_length

            if shift_length < 12:   #12
                # print xy2.shape,min_index,len(distance)
                x1_paired.append(temp_x[min_index] - temp_u[min_index])
                y1_paired.append(temp_y[min_index] - temp_v[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + temp_u[min_index])
                v.append(v_temp + temp_v[min_index])

                del temp_x[min_index], temp_y[min_index], temp_u[
                    min_index], temp_v[min_index]

                if shift_length > 10:   #10
                    refresh = True

        return x1_paired, y1_paired, x2_paired, y2_paired, u, v, refresh

    def dispOpticalFlow(self, im_cal, x, y, u, v, theta, slip_indicator):
        # mask = np.zeros_like(im_cal)
        mask2 = np.zeros_like(im_cal)
        amf = 1
        slip_scale = 5
        x = np.array(x).astype(np.int16)
        y = np.array(y).astype(np.int16)
        for i in range(u.shape[0]):  # self.u_sum

            mask2 = cv2.line(mask2,
                             (int(x[i] + u[i] * amf), int(y[i] + v[i] * amf)),
                             (x[i], y[i]), [0, 120, 120], 2)

        img = cv2.add(im_cal / 1.5, mask2)

        if slip_indicator:
            img = img + self.im_slipsign / 2
            cv2.arrowedLine(img, (60, 75), (int(60 + slip_scale * np.mean(u)), int(75 + slip_scale * np.mean(v))),
                            (255, 255, 0), 3, tipLength=0.2)
            if theta > 10:
                cv2.putText(img, 'CounterClockwise', (6, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                            cv2.LINE_AA)
            if theta < -10:
                cv2.putText(img, 'Clockwise', (6, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                            cv2.LINE_AA)



        return img.astype(np.uint8)

    def preprocess(self, img, ref):
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1
        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp2 = img_smooth - blur
        diff_temp3 = np.clip((diff_temp2 - self.zeropoint) / self.lookscale, 0, 0.999)
        diff = (diff_temp3 * self.bin_num).astype(int)

        return diff

    def matching(self, img, ref, table):
        diff = self.preprocess(img, ref)
        grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
        return grad_img


    def arcsin_and_arccos(self, pt2):
        pt1 = [0, 0]
        delta_x = pt2[0] - pt1[0]
        delta_y = pt2[1] - pt1[1]
        sin = delta_y / math.sqrt(delta_x ** 2 + delta_y ** 2)
        cos = delta_x / math.sqrt(delta_x ** 2 + delta_y ** 2)
        if sin >= 0 and cos >= 0:
            return math.asin(sin), math.acos(cos)
        elif sin >= 0 and cos < 0:
            return math.pi - math.asin(sin), math.acos(cos)
        elif sin < 0 and cos < 0:
            return math.pi - math.asin(sin), 2 * math.pi - math.acos(cos)
        elif sin < 0 and cos >= 0:
            return 2 * math.pi + math.asin(sin), 2 * math.pi - math.acos(cos)

    def videoloop(self):
        try:
            while not self.stopEvent.is_set():
                _, self.frame = self.vs.read()
                frame_show = self.frame.copy()
                # print(f"fps is {vs.get(cv2.CAP_PROP_FPS)}")
                # self.frame = cv2.medianBlur(self.frame, 3)
                # y = 80
                # x = 170
                # crop_img = self.frame[y:y+390, x:x+390]
                #crop_img = imutils.rotate(crop_img, 0)
                #self.frame = imutils.resize(self.frame, width=390)
                # self.frame = cv2.undistort(self.frame, self.cameraMatrix, self.disCoeffs, None)

                output = self.pose_esitmation(self.frame, self.aruco_dict_type)

                if self.idcheck is False:
                    self.tvec = self.pre_tvec

                if self.resetsensor:
                    self.ati_bias = self.ATIdata.copy()
                    self.sensor_bias = self.tvec.copy()
                    self.resetsensor = False

                self.ATIdata = self.ATIdata - self.ati_bias
                self.tvec = self.tvec - self.sensor_bias
                # np.transpose
                self.ATIdata = np.matmul(self.FT_Cali, self.ATIdata)
                self.tvecFT = np.matmul(self.Sensor_Cali, self.tvec)

                Alldata = np.vstack((self.ATIdata[:3], self.tvecFT))

                self.frame_Gel = self.frame[self.FrameGel_iniy:self.FrameGel_iniy+self.FrameGel_height,
                                 self.FrameGel_inix:self.FrameGel_inix+self.FrameGel_width]

                marker_mask_new = np.zeros_like(self.frame_Gel)
                inpaint_image = np.zeros_like(self.frame_Gel)
                raw_image = np.zeros_like(self.frame_Gel)
                slip_img = np.zeros_like(self.frame_Gel)
                slip_img2 = np.zeros_like(self.frame_Gel)
                depth = np.zeros_like(self.frame_Gel)
                diff = np.zeros_like(self.frame_Gel)
                marker_new = np.zeros_like(self.frame_Gel)
                u_sum, v_sum, uv_sum = .0, .0, .0
                x, y = math.pi, math.pi
                theta, theta1= 0, 0

                if self.con_flag1:
                    # self.frame_Gel = self.frame_Gel[self.x_index, self.y_index, :]

                    ref_image = self.frame_Gel.copy()     # select which ref to use
                    # ref_image = cv2.imread("./ref.jpg")

                    ref_image = ref_image[self.x_index, self.y_index, :]

                    imgwc = np.array(ref_image).astype(np.float32)
                    self.im_slipsign = np.zeros(imgwc.shape)
                    cv2.putText(self.im_slipsign, 'Slip!', (35, 35),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2,
                                cv2.LINE_AA)
                    im_gray = self.rgb2gray(imgwc).astype(np.uint8)
                    self.dmask1 = self.defect_mask(im_gray)
                    marker = self.creat_mask_2(imgwc, self.dmask1)
                    # cv2.imwrite("./ref.jpg", ref_image)

                    # marker = self.marker_detection(ref_image.copy())
                    keypoints = self.find_dots(marker)
                    if self.reset_shape1:
                        marker_mask = self.make_mask(ref_image.copy(), keypoints)
                        ref_image = cv2.inpaint(ref_image, marker_mask, 3, cv2.INPAINT_TELEA)
                        self.u_sum1 = np.zeros(len(keypoints))
                        self.v_sum1 = np.zeros(len(keypoints))
                        self.u_addon1 = list(self.u_sum1)
                        self.v_addon1 = list(self.v_sum1)
                        self.x1_last1 = []
                        self.y1_last1 = []
                        for i in range(len(keypoints)):
                            self.x1_last1.append(keypoints[i].pt[0] / self.scale)
                            self.y1_last1.append(keypoints[i].pt[1] / self.scale)
                        self.x_iniref1 = list(self.x1_last1)
                        self.y_iniref1 = list(self.y1_last1)

                    marker_num = len(keypoints)
                    # print(f"marker num is {marker_num}")
                    # mp_array = np.zeros((marker_num, 3, 200))
                    # index_ref = np.linspace(0, marker_num - 1,
                    #                         marker_num).astype(int)

                    keypoints = self.sortkeypoints(keypoints)
                    self.con_flag1 = False
                    self.reset_shape1 = False
                    self.absmotion1 = 0
                    self.absmotion2 = 0

                else:
                    if self.restart1:
                        self.con_flag1 = True
                        self.restart1 = False
                        self.absmotion1 = 0
                        self.absmotion2 = 0
                        self.slip_indicator1 = False
                        self.slip_indicator2 = False
                    # self.frame_Gel = self.frame_Gel[self.x_index, self.y_index, :]
                    raw_image = self.frame_Gel.copy()

                    # cv2.imwrite("./raw.jpg", raw_image)

                    raw_image = cv2.GaussianBlur(self.frame_Gel, (3, 3), 0).astype(np.uint8)
                    raw_image = raw_image[self.x_index, self.y_index, :]

                    im_cal_show = np.array(raw_image.copy()).astype(np.float32)

                    # marker_new = self.marker_detection(raw_image.copy())
                    # marker_new = marker_new * self.dmask1
                    marker_new = self.creat_mask_2(raw_image.copy(), self.dmask1)

                    keypoints_new = self.find_dots(marker_new)
                    marker_mask_new = self.make_mask(raw_image.copy(), keypoints_new)
                    keypoints_new = self.sortkeypoints(keypoints_new)
                    inpaint_image = cv2.inpaint(raw_image, marker_mask_new, 3,
                                            cv2.INPAINT_TELEA)

                    # self.m.init(keypoints_new)
                    # self.m.run()
                    # flow = self.m.get_flow()
                    # self.draw_flow(raw_image, flow)

                    if self.useTable is True:
                        grad_img2 = self.matching(inpaint_image, ref_image, self.table)
                        depth = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])
                        depth -= 0.2
                        depth[depth < 0] = 0
                        # depth = (depth * 200).astype(np.uint8)
                        depth = cv2.applyColorMap((depth * 200).astype(np.uint8), cv2.COLORMAP_BONE)

                    if self.slip is True:
                        if self.refresh1:
                            keypoints_new1 = self.find_dots(marker_new)
                            keypoints_new1 = np.asarray(self.sortkeypoints(keypoints_new1))
                            x1, y1, x2, y2, u, v = self.flow_calculate_global_1(keypoints_new1)
                            self.refresh1 = False
                        else:
                            keypoints_new1 = self.find_dots(marker_new)
                            keypoints_new1 = np.asarray(self.sortkeypoints(keypoints_new1))
                            x1, y1, x2, y2, u, v, self.refresh1 = self.flow_calculate_in_contact_1(keypoints_new1)

                        x2_center = np.expand_dims(np.array(x2), axis=1)
                        y2_center = np.expand_dims(np.array(y2), axis=1)
                        x1_center = np.expand_dims(np.array(x1), axis=1)
                        y1_center = np.expand_dims(np.array(y1), axis=1)
                        p2_center = np.expand_dims(np.concatenate((x2_center, y2_center), axis=1), axis=0)
                        p1_center = np.expand_dims(np.concatenate((x1_center, y1_center), axis=1), axis=0)
                        tran_matrix = cv2.estimateRigidTransform(p1_center, p2_center, False)
                        # theta = np.arctan(-tran_matrix[0, 1] / tran_matrix[0, 0]) * 180. * np.pi
                        # theta = np.arctan(-tran_matrix[0, 1] / tran_matrix[0, 0]) * 180 * np.pi

                        theta = 0

                        u_sum, v_sum, uv_sum = np.array(u), np.array(v), np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)
                        self.absmotion1 = np.mean(uv_sum)

                        if max(np.abs(np.mean(u_sum)), np.abs(np.mean(v_sum)), np.abs(np.mean(uv_sum))) > 0.5:
                            self.collision_detected = True

                        self.slip_indicator1 = max(
                            np.abs(np.mean(u_sum)), np.abs(np.mean(v_sum)),
                            np.abs(np.mean(uv_sum))) > self.collideThre or abs(
                            theta) > self.collide_rotation

                        if self.showimage1:
                            # self.slip_indicator1 = False
                            slip_img = self.dispOpticalFlow(im_cal_show, x2, y2, u_sum, v_sum, theta, self.slip_indicator1)
                            # print(f"refresh tag is {self.refresh1}")

                        slip_vec = [np.mean(u_sum), np.mean(v_sum)]
                        x, y = self.arcsin_and_arccos(slip_vec)

                FTbackground = np.zeros([200, 630, 3], dtype=np.uint8)
                FTbackground.fill(105)
                R1, R2, R3 = 255, 255, 255
                G1, G2, G3 = 0, 0, 0

                if self.tvec[0] > 0:
                    R1 = 255
                    G1 = 0
                elif self.tvec[0] < 0:
                    R1 = 0
                    G1 = 255

                if self.tvec[1] > 0:
                    R2 = 255
                    G2 = 0
                elif self.tvec[1] < 0:
                    R2 = 0
                    G2 = 255

                if self.tvec[2] > 0:
                    R3 = 255
                    G3 = 0
                elif self.tvec[2] < 0:
                    R3 = 0
                    G3 = 255

                font = cv2.FONT_HERSHEY_DUPLEX
                FTbackground = cv2.putText(FTbackground, 'Fx:', (10, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                FTbackground = cv2.putText(FTbackground, 'Fy:', (10, 83), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                FTbackground = cv2.putText(FTbackground, 'Fz:', (10, 126), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

                FTbackground = cv2.rectangle(FTbackground, (70, 18), (70 + abs(int(10 * self.tvecFT[0])), 45), (R1, G1, 0), -1)  # 50000
                FTbackground = cv2.rectangle(FTbackground, (70, 61), (70 + abs(int(10 * self.tvecFT[1])), 88), (R2, G2, 0), -1)
                FTbackground = cv2.rectangle(FTbackground, (70, 104), (70 + abs(int(10 * self.tvecFT[2])), 131), (R3, G3, 0), -1)

                image = imutils.resize(self.frame, width=240) #slip_img frame_show
                imageGray = imutils.resize(slip_img, width=120)    #depth slip_img2
                imageAll = imutils.resize(depth, width=120)

                image = Image.fromarray(image)  # self.frame_Gel output
                imageGray = Image.fromarray(imageGray)     # output raw_img inpaint_image
                imageAll = Image.fromarray(imageAll)
                imageFT = Image.fromarray(FTbackground)

                image = ImageTk.PhotoImage(image)
                imageGray = ImageTk.PhotoImage(imageGray)
                imageAll = ImageTk.PhotoImage(imageAll)
                imageFT = ImageTk.PhotoImage(imageFT)

                if self.panel1 is None or self.panel2 is None or self.panel3 is None or self.label1 is None:
                    # self.panel1 = tk.Label(image=image)
                    self.panel1.image = image
                    # self.panel1.place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.4)

                    # self.panel2 = tk.Label(image=imageGray)
                    self.panel2.image = imageGray
                    # self.panel2.place(relx=0.55, rely=0.1, relwidth=0.4, relheight=0.4)
                    self.panel3.image = imageAll

                    self.label1.image = imageFT

                else:
                    self.panel1.configure(image=image)
                    self.panel1.image = image
                    self.panel2.configure(image=imageGray)
                    self.panel2.image = imageGray
                    self.panel3.configure(image=imageAll)
                    self.panel3.image = imageAll
                    self.label1.configure(image=imageFT)
                    self.label1.image = imageFT

                # self.label2['text'] = 'u_sum' + str(math.degrees(x)) + '\n' + 'v_sum' + str(math.degrees(y)) + '\n' + str(theta)
                # self.label2['text'] = 'theta is' + str(theta) + '\n' + str(self.refresh1) + '\n theta1 is ' + str(theta1)
                self.label2['text'] = 'Force (N):' + '\n' + str(self.tvecFT) + '\n' + 'Depth (mm) is '+ str(np.max(depth)/200)
                # self.label2['text'] = str(str(self.tvec))

                Objectsdata = np.vstack((self.tvecFT, np.max(depth)/200))

                path_file = 'D:/PyCharmProject/TactileSensor/TestWifiCamera/GelForceWireless/'
                if self.writedata:
                    print("Saving FT...")
                    with open(path_file + 'OurSensorDataRange.csv', 'a', newline='') as csvfile:
                        datawriter = csv.writer(csvfile)

                        # datawriter.writerow(np.vstack((self.counter,Alldata)).flatten())
                        datawriter.writerow(np.vstack((self.counter, Objectsdata)).flatten())

                    # with open(path_file + 'thetaData.csv', 'a', newline='') as csvfile:
                    #     datawriter = csv.writer(csvfile)
                    #     datawriter.writerow(np.vstack((self.counter, theta1)).flatten())

                if self.writeTacdata:
                    print("Saving Tac...")
                    # cv2.imwrite(path_file + '/TacData/Sample' + str(self.taccount).zfill(2) + '.jpg', self.frame_Gel)

                    # cv2.imwrite(path_file + '/TacData_raw/Sample'+str(self.taccount).zfill(2)+'.jpg', raw_image)
                    # cv2.imwrite(path_file + '/TacData_fill/Sample'+str(self.taccount).zfill(2)+'.jpg', inpaint_image)
                    cv2.imwrite(path_file + '/TacData_fill/Slip/Slip' + str(self.taccount).zfill(3) + '.jpg', self.frame)
                    cv2.imwrite(path_file + '/TacData_fill/Slip/Depth' + str(self.taccount).zfill(3) + '.jpg', depth)
                    cv2.imwrite(path_file + '/TacData_fill/Slip/Tactile' + str(self.taccount).zfill(3) + '.jpg', slip_img)
                    self.taccount += 1
                    # self.writeTacdata = not self.writeTacdata

                self.counter = self.counter + 1


        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def startsensor(self):
        self.resetsensor = True
        self.restart1 = True
        self.reset_shape1 = True

    def plotdata(self):
        self.writedata = not self.writedata

    def quitsensor(self):
        # self.writeTacdata = True
        self.writeTacdata = not self.writeTacdata
        # self.root.destroy()
        # self.root.quit()
        # exit()

    def onclose(self):
        print("[INFO] Closing...")
        self.stopEvent.set()
        self.vs.release()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    print("[INFO] warming up webcam...")

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]

    # vs = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    # vs = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    vs = cv2.VideoCapture("rtsp://192.168.3.87:8554/mjpeg/1")
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 480) #640
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 320) #480
    vs.set(cv2.CAP_PROP_FPS, 20)
    time.sleep(2.0)

    pba = Processing(vs, aruco_dict_type)
    pba.root.mainloop()