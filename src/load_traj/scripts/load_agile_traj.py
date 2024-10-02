#!/usr/bin/env python3

import csv
import os
import numpy as np
from aims_msgs.msg import ReferenceTrajectory
import rospy

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header
        data = [row for row in csv_reader]
    return header, data

def publish_trajectory(states, time_steps, controls):
    rospy.init_node('trajectory_publisher', anonymous=True)
    pub = rospy.Publisher('/trajectory', ReferenceTrajectory, queue_size=10)

    trajectory_msg = ReferenceTrajectory()
    trajectory_msg.traj_name = "scvx_traj1"
    trajectory_msg.seq_len = states.shape[0]
    trajectory_msg.trajectory = np.reshape(states, (-1, )).tolist()
    trajectory_msg.time = time_steps.tolist()
    trajectory_msg.inputs = np.reshape(controls, (-1, )).tolist()

    # Keep publishing the trajectory
    # rate = rospy.Rate(1)  # 1 Hz
    # while not rospy.is_shutdown():
    #     pub.publish(trajectory_msg)
    #     rate.sleep()

    # Publishing only once
    rospy.sleep(1)
    pub.publish(trajectory_msg)

if __name__ == "__main__":
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    file_path = current_folder_path+'/scvx_traj1.csv'
    header, data = read_csv(file_path)
    data = np.array(data, dtype=float)
    data = data[:, 1:] # Skip the first column

    states = data[:, :13]
    time_steps = data[:, -1]
    controls = data[:, -5:-1]

    publish_trajectory(states, time_steps, controls)
