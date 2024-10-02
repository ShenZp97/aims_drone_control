#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import message_filters
from scipy.spatial.transform import Rotation as R


def callback(pose_sub, twist_sub, target_pose_sub, target_twist_sub):

    target_p = target_pose_sub.pose.position
    target_q = target_pose_sub.pose.orientation
    target_v = target_twist_sub.twist.linear
    drone_p = pose_sub.pose.position
    drone_q = pose_sub.pose.orientation
    drone_v = twist_sub.twist.linear
    drone_w = twist_sub.twist.angular


    r = R.from_quat([target_q.x, target_q.y, target_q.z, target_q.w]) # scalar-last
    r_drone = R.from_quat([drone_q.x, drone_q.y, drone_q.z, drone_q.w]) # scalar-last

    r_euler = r.as_euler('zyx', degrees=False) # euler angles
    t_yaw = -r_euler[0] # yaw angle
    # r_matrix_yaw = np.array([[np.cos(t_yaw), -np.sin(t_yaw), 0.],
    #                          [np.sin(t_yaw), np.cos(t_yaw), 0.],
    #                          [0., 0., 1.]]) # rotation matrix only for yaw
    r_matrix_yaw = R.from_euler('z', t_yaw, degrees=False) # rotation matrix only for yaw
    
    p_relative = np.array([drone_p.x-target_p.x,
                           drone_p.y-target_p.y,
                           drone_p.z-target_p.z])
    v_relative = np.array([drone_v.x-target_v.x,
                           drone_v.y-target_v.y,
                           drone_v.z-target_v.z])

    # p_relative = np.matmul(r_matrix_yaw, p_relative) # rotate yaw
    # v_relative = np.matmul(r_matrix_yaw, v_relative)
    p_relative_true = r_matrix_yaw.apply(p_relative) # rotate yaw
    v_relative_true = r_matrix_yaw.apply(v_relative)
    
    drone_euler = r_drone.as_euler('zyx', degrees=False) # drone attitude
    drone_euler[0] = drone_euler[0] - np.arctan(p_relative[1]/p_relative[0])
    drone_q_real = R.from_euler('zyx', drone_euler, degrees=False)
    drone_q_real = drone_q_real.as_quat()

    odom.header = pose_sub.header
    odom.pose.pose.position.x = p_relative_true[0]
    odom.pose.pose.position.y = p_relative_true[1]
    odom.pose.pose.position.z = p_relative_true[2]
    odom.pose.pose.orientation.x = drone_q_real[0]
    odom.pose.pose.orientation.y = drone_q_real[1]
    odom.pose.pose.orientation.z = drone_q_real[2]
    odom.pose.pose.orientation.w = drone_q_real[3]

    odom.twist.twist.linear.x = v_relative_true[0]
    odom.twist.twist.linear.y = v_relative_true[1]
    odom.twist.twist.linear.z = v_relative_true[2]
    odom.twist.twist.angular = drone_w

    pub.publish(odom)
    rate.sleep()


rospy.init_node('vicon_relative')
pub = rospy.Publisher('/vrpn_client_node/aims3/odom', Odometry, queue_size=1)
odom = Odometry()

target_pose_sub = message_filters.Subscriber('/vrpn_client_node/aims_ugv/pose', PoseStamped)
target_twist_sub = message_filters.Subscriber('/vrpn_client_node/aims_ugv/twist', TwistStamped)
pose_sub = message_filters.Subscriber('/vrpn_client_node/aims3/pose', PoseStamped)
twist_sub = message_filters.Subscriber('/vrpn_client_node/aims3/twist', TwistStamped)

ts = message_filters.ApproximateTimeSynchronizer([pose_sub, twist_sub, target_pose_sub, target_twist_sub],
                                                1, 1)
ts.registerCallback(callback)

rate = rospy.Rate(200)


rospy.spin()