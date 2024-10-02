#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import message_filters

def callback(pose_sub, twist_sub):
    odom.header = pose_sub.header
    odom.pose.pose = pose_sub.pose
    odom.twist.twist = twist_sub.twist

    pub.publish(odom)
    rate.sleep()


rospy.init_node('vicon2odom')
pub = rospy.Publisher('/vrpn_client_node/flywoo75/odom', Odometry, queue_size=1)
odom = Odometry()

pose_sub = message_filters.Subscriber('/vrpn_client_node/flywoo75/pose', PoseStamped)
twist_sub = message_filters.Subscriber('/vrpn_client_node/flywoo75/twist', TwistStamped)

ts = message_filters.ApproximateTimeSynchronizer([pose_sub, twist_sub], 1, 1)
ts.registerCallback(callback)

rate = rospy.Rate(200)


rospy.spin()