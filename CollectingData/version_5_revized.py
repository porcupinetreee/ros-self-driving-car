#!/usr/bin/env python
import cv2
import math
import numpy as np
# from matplotlib import pyplot as plt
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def make_points(frame, line):
    height = frame.shape[0]
    width = frame.shape[1]
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        print('No line_segment segments detected')
        return lane_lines

    height = frame.shape[0]
    width = frame.shape[1]
    left_fit = []
    right_fit = []

    boundary = 1 / 2
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if x1 < left_region_boundary and x2 < left_region_boundary:
                left_fit.append((slope, intercept))

            if x1 > right_region_boundary and x2 > right_region_boundary:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    return lane_lines


def draw_lines(img, lines):
    img = np.copy(img)
    blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    h = img.shape[0]
    w = img.shape[1]

    left_x1, left_y1, left_x2, left_y2 = lines[0][0]
    right_x1, right_y1, right_x2, right_y2 = lines[1][0]
    # print(right_x2)
    # print(right_x1)
    print("   ")
    midx = int(w / 2)
    midy = int(h / 2)
    if left_x1 <= 0:
        left_x1 = 0
    if left_x2 <= 0:
        left_x2 = 0
    if right_x1 <= 0:
        right_x1 = 0
    if right_x2 <= 0:
        right_x2 = 0
    ave_x2 = (right_x2 + left_x2) / 2
    ave_x1 = (right_x1 + left_x1) / 2
    x = ave_x2 - ave_x1
    steering_angle = math.atan(x / 300.0)
    steering_angle = steering_angle * 180 / math.pi
    print(steering_angle)
    return steering_angle


# image = cv2.imread('calib_duz.jpg')
# image = cv2.resize(image,(800,600))
# cv2.imshow('raw image',image)
pts1 = np.array([(60, 334.56), (774.5, 334.56), (0,600), (800, 600)], dtype="float32")
pts2 = np.array([(0, 0), (800, 0), (0, 600), (800, 600)], dtype="float32")
M = cv2.getPerspectiveTransform(pts1, pts2)

bridge = CvBridge()

cap = cv2.VideoCapture(0)
rospy.init_node('Pi_camera', anonymous=True)
while (cap.isOpened()):
    
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        scale_percent = 60  # percent of original size
        width = 800
        height = 600
        dim = (width, height)
        # cv2.imshow('raw camera view', frame)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = cv2.warpPerspective(frame, M, (800, 600))
        cv2.imshow('wrapped camera view', frame)
        # plt.imshow(frame)
        # plt.show()
        frame = cv2.blur(frame, (5, 5))

	
	frame2 = cv2.resize(frame, (32,32), interpolation=cv2.INTER_AREA)
	image_message = bridge.cv2_to_imgmsg(frame2, encoding="passthrough")
	pub2 = rospy.Publisher('ros_image', Image, queue_size=10)
	pub2.publish(image_message)

        frame = cv2.Canny(frame, 100, 200)
        height, width = frame.shape
        mask = np.zeros_like(frame)
        polygon = np.array([[
            (0, 300), (800, 300), (800, 600), (0, 600),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        frame = cv2.bitwise_and(frame, mask)
        # cv2.imshow('cropped', frame)
        
	try:
		lines = cv2.HoughLinesP(frame, rho=3, theta=np.pi / 180, threshold=120, lines=np.array([]), minLineLength=1, maxLineGap=1000)
        	lane_lines = average_slope_intercept(frame, lines)
        	angle = draw_lines(frame, lane_lines)
        	pub = rospy.Publisher('steering_angle', Int16, queue_size=10)
        	pub.publish(angle)
	

   	except:
		pass
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

