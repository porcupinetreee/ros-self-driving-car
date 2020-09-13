#!/usr/bin/env python
from getch import getch
import rospy
from std_msgs.msg import String


def key_in():
    pub = rospy.Publisher('key_press', String, queue_size=100)
    rate = rospy.Rate(10)
    angle = 0
    while not rospy.is_shutdown():
        key = getch()
        if key == 'w':
            rospy.loginfo(key)
            pub.publish(key)
        elif key == 'a':
            rospy.loginfo(key)
            pub.publish(key)
        elif key == 's':
            rospy.loginfo(key)
            pub.publish(key)
        elif key == 'd':
            rospy.loginfo(key)
            pub.publish(key)
        
        rate.sleep()

if __name__== '__main__':
    try:
        rospy.init_node('talker', anonymous=True)
        print('Key_press node has been activated. Enter keys down here->')
        key_in()
    except rospy.ROSInterruptException:
        pass
