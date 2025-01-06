#!/usr/bin/env python3

import csv
import os
import numpy as np
from aims_msgs.msg import ReferenceTrajectory
import rospy
import argparse


def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]
    return data

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
    rospy.sleep(0.2)
    pub.publish(trajectory_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPC Trajectory Publisher Node")
    parser.add_argument("-f", "--file", type=str, default="new_scvx_traj.csv", help="trajectory file")
    args = parser.parse_args()

    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    file_path = current_folder_path+"/"+args.file
    data = read_csv(file_path)
    data = np.array(data, dtype=float)
    data = data.T

    x_traj = data[:, :15]
    u_traj = data[:, 15:]
    states = x_traj[:, :13]
    time_steps = x_traj[:, -2]
    controls = u_traj[:, :-1]

    publish_trajectory(states, time_steps, controls)
