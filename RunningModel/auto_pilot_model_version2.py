#!/usr/bin/env python
import rospy
import tensorflow as tf
import tflearn
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from alexnet import alexnet
from cv_bridge import CvBridge


WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = MODEL_NAME = 'self-driving-car-BALANCED-{}-{}-{}.model'.format(LR, 'alexnet', EPOCHS)
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

rospy.init_node('auto_pilot_model', anonymous=True)
pub = rospy.Publisher('key_press', String, queue_size=10)
bridge = CvBridge()

def callback_pub(img):
    cv_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image = cv2.resize(cv_image,(WIDTH,HEIGHT))
    prediction = model.predict([cv_image.reshape(WIDTH,HEIGHT,1)])[0]
    moves = list(np.around(prediction))
    if moves == [0, 1, 0]:
        pub.publish('w')
    elif moves == [1, 0, 0]:
        pub.publish('a')
    elif moves == [0, 0, 1]:
        pub.publish('d')
    elif moves == [0, 0, 0]:
        pub.publish('s')
    


def initialize():
    rospy.Subscriber("ros_image", Image, callback_pub)
    rospy.spin()
    
if __name__ == '__main__':
    try:
		initialize()
    except rospy.ROSInterruptException:
        	pass

