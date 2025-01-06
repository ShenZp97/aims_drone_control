#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from controller import Robot
import os
from nav_msgs.msg import Odometry
import numpy as np
import yaml

from std_msgs.msg import String  # For sending commands 
from geometry_msgs.msg import Point, PoseStamped  # For positions
from aims_msgs.msg import ReferenceTrajectory

from DroneLQR import LQRController
from mpc_drone_optimizer import QuadrotorParam
from mpc_drone import Quad3DMPC

import os 

# def callback(data):
#     global velocity
#     global m essage
#     message = 'Received velocity value: ' + str(data.data)
#     velocity = data.data


# robot = Robot()
# timeStep = int(robot.getBasicTimeStep())
# left = robot.getDevice('motor.left')
# right = robot.getDevice('motor.right')
# sensor = robot.getDevice('prox.horizontal.2')  # front central proximity sensor
# sensor.enable(timeStep)
# left.setPosition(float('inf'))  # turn on velocity control for both motors
# right.setPosition(float('inf'))
# velocity = 0
# left.setVelocity(velocity)
# right.setVelocity(velocity)
# message = ''
# print('Initializing ROS: connecting to ' + os.environ['ROS_MASTER_URI'])
# robot.step(timeStep)
# rospy.init_node('listener', anonymous=True)
# print('Subscribing to "motor" topic')
# robot.step(timeStep)
# rospy.Subscriber('motor', Float64, callback)
# pub = rospy.Publisher('sensor', Float64, queue_size=10)
# print('Running the control loop')
# while robot.step(timeStep) != -1 and not rospy.is_shutdown():
#     pub.publish(sensor.getValue())
#     print('Published sensor value: ', sensor.getValue())
#     if message:
#         print(message)
#         message = ''
#     left.setVelocity(velocity)
#     right.setVelocity(velocity)

class MavicDriver:
    def __init__(self):
        self.__robot = Robot()
        self.__timestep = int(self.__robot.getBasicTimeStep())
        # Sensors
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__imu = self.__robot.getDevice('inertial unit')
        self.__camera = self.__robot.getDevice('camera')
        self.__camera.enable(2*self.__timestep)
        self.__gps.enable(self.__timestep)
        self.__gyro.enable(self.__timestep)
        self.__imu.enable(self.__timestep)

        # Propellers
        self.__propellers = [
            self.__robot.getDevice('front right propeller'),
            self.__robot.getDevice('front left propeller'),
            self.__robot.getDevice('rear right propeller'),
            self.__robot.getDevice('rear left propeller')
        ]

        for propeller in self.__propellers: # Use velocity command
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)

        self.simtime = 0.0

        #-----FSM
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path+'/config/mavic_config.yaml','r') as f:
            config = yaml.safe_load(f)
        self.mass = config['mass']
        self.warm_start_option = config['warm_start_option']
        # self.lqrcontroller = LQRController(config)
        drone_config = QuadrotorParam(config)
        self.mpc_controller = Quad3DMPC(drone_config,
                                        r_cost=config['r_cost'],
                                        q_cost=config['q_cost'])
        
        self.state = "waiting command"
        self.tracking = False
        self.reach_tolerance = 0.3
        
        self.odom = None
        self.takeoff_height = config['takeoff_h']
        self.control_hz = config["control_hz"] # control frequency
        self.odom_frd = False # Is the frame of odom a frd frame?
        self.cmd_frd = False # Is the frame of cmd a frd frame?
        self.hover_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.goto_p =  np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.landing_his = np.ones(self.control_hz) # landing history for ground detection
        self.landing_dis = 1.0 # point distance for landing phase
        self.mission_finished = False
        self.waiting_pos = np.array([0.0, 0.0, 0.0])
        # self.traj = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        self.n_state = 13
        self.n_control = 4

        self.tracking_time = 0.0


        #ROS
        rospy.init_node('mavic_driver', anonymous=False)

        self.command_subscriber = rospy.Subscriber('drone_command', String, self.handle_command)
        self.goto_p_subscriber = rospy.Subscriber('goto_position', PoseStamped, self.update_target_position)
        self.traj_subscriber = rospy.Subscriber('trajectory', ReferenceTrajectory, self.traj_callback)

        self.odom_msg = Odometry()
        self.odom_pub = rospy.Publisher('mavic_odom', Odometry, queue_size=10)
        rospy.loginfo('Controller initialized')

        while self.__robot.step(self.__timestep) != -1 and not rospy.is_shutdown():
            self.simtime += self.__timestep
        
            q = np.array(self.__imu.getQuaternion())
            pos = np.array(self.__gps.getValues())
            w = np.array(self.__gyro.getValues())
            vel = np.array(self.__gps.getSpeedVector())
            q = np.array([q[3], q[0], q[1], q[2]])

            # FRD to FLU
            if self.odom_frd:
                rospy.logwarn("Webots should use FLU frame")
                # p_flu = np.array([pos.x, -pos.y, -pos.z])
                # q_flu = np.array([q.w, q.x, -q.y, -q.z])
                # Q = self.lqrcontroller.qtoQ(q_flu) # body to inertial
                # v_flu = np.array([v.x, -v.y, -v.z])
                # v_b = Q.T @ v_flu
                # w_flu = np.array([w.x, -w.y, -w.z])
            else:
                p_flu = pos
                q_flu = q
                # Q = self.lqrcontroller.qtoQ(q_flu) # body to inertial
                v_flu = vel
                # v_b = Q.T @ v_flu
                w_flu = w

            # Publish Odom
            self.odom_msg.pose.pose.position.x = p_flu[0]
            self.odom_msg.pose.pose.position.y = p_flu[1]
            self.odom_msg.pose.pose.position.z = p_flu[2]
            self.odom_msg.pose.pose.orientation.w = q_flu[0]
            self.odom_msg.pose.pose.orientation.x = q_flu[1]
            self.odom_msg.pose.pose.orientation.y = q_flu[2]
            self.odom_msg.pose.pose.orientation.z = q_flu[3]
            self.odom_msg.twist.twist.linear.x = v_flu[0]
            self.odom_msg.twist.twist.linear.y = v_flu[1]
            self.odom_msg.twist.twist.linear.z = v_flu[2]
            self.odom_msg.twist.twist.angular.x = w_flu[0]
            self.odom_msg.twist.twist.angular.y = w_flu[1]
            self.odom_msg.twist.twist.angular.z = w_flu[2]
            self.odom_pub.publish(self.odom_msg)


            # state = np.hstack((p_flu, q_flu, v_b, w_flu)) # converted to FLU
            state_vi = np.hstack((p_flu, q_flu, v_flu, w_flu)) # converted to FLU. Not body velocity
            self.odom = state_vi

            if self.state == "trajectory tracking":
                if not self.tracking:
                    if np.linalg.norm(self.x_ref[0, 0:3] - p_flu) > self.reach_tolerance:
                        x_des = np.hstack((self.x_ref[0, 0:3], np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)))
                        rospy.loginfo_once("Raising drone to the start point of the trajectory...")
                    else:
                        self.tracking_time = self.simtime/1e3
                        self.tracking = True

                if self.tracking:
                    # if np.linalg.norm(self.x_ref[-1, 0:3]-p_flu) <= self.reach_tolerance: # reached end of trajectory
                    if np.linalg.norm(self.x_ref[-1, 0:3]-p_flu) <= self.reach_tolerance and (self.simtime/1e3 - self.tracking_time)>=1.0: # reached end of trajectory and trajectory time is more than 1s
                        self.hover_pose = self.x_ref[-1, 0:7]
                        self.state = "hover"
                        self.tracking = False
                        rospy.loginfo("Tracking finished. Hover...")
                    else:
                        
                        # total_index = x_ref.shape[0]-1
                        # Trajectory tracking

                        # use state to get reference
                        # current_idx = np.argmin(np.linalg.norm(self.x_ref[:, 0:3]-p_flu, axis=1))
                        # t_ref_now = self.t_ref[current_idx]

                        # use time to get reference
                        t_ref_now = self.simtime/1e3 - self.tracking_time

                        t_ref_segment = np.linspace(t_ref_now, t_ref_now + self.mpc_controller.t_horizon, num=self.mpc_controller.n_nodes + 1)

                        x_ref_mpc = np.zeros((self.mpc_controller.n_nodes + 1, self.n_state))
                        u_ref_mpc = np.zeros((self.mpc_controller.n_nodes + 1, self.n_control))

                        for i in range(self.n_state):
                            x_ref_mpc[:, i] = np.interp(t_ref_segment, self.t_ref, self.x_ref[:, i])

                        for i in range(self.n_control):
                            u_ref_mpc[:, i] = np.interp(t_ref_segment, self.t_ref, self.u_ref[:, i])

            elif self.state=="takeoff":
                if np.abs(p_flu[2] - self.takeoff_height)<=self.reach_tolerance: # reached height
                    p_takeoff = p_flu.copy()
                    p_takeoff[2] = self.takeoff_height
                    self.hover_pose[0:3] = p_takeoff
                    self.hover()
                    
                else:
                    p_takeoff = p_flu.copy()
                    p_takeoff[2] = self.takeoff_height
                    x_des = np.hstack((p_takeoff, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)))

            elif self.state=="gotopoint":
                if np.linalg.norm(p_flu - self.goto_p[0:3])<=self.reach_tolerance: # reached point
                    self.hover_pose = self.goto_p
                    self.hover()
                    
                else:
                    x_des = np.hstack((self.goto_p, np.zeros(3), np.zeros(3)))

            elif self.state=="land":
                p_land = p_flu.copy()
                p_land[2] = p_land[2]-self.landing_dis
                self.landing_his[:-1] = self.landing_his[1:] # pop history
                self.landing_his[-1] = p_land[2] # pop history
                if (np.max(self.landing_his)-np.min(self.landing_his))<=0.1: # reached floor
                    self.waiting_command()
                    self.mission_finished = True
                            
                else:
                    x_des = np.hstack((p_land, np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)))

            if self.state=="hover":
                x_des = np.hstack((self.hover_pose, np.zeros(3), np.zeros(3)))

            # elapsed_time = (time() - self.start_time)/1e3
            # print("elapsed time:", elapsed_time)

            # if elapsed_time >= 1/self.control_hz: # make controller run at the desired hz
            #     self.start_time = time()
            if self.state == "waiting command": # do nothing
                cmd = np.zeros(4)
            elif self.state == "trajectory tracking" and self.tracking: # use MPC
                # rospy.loginfo("MPC tracking.")
                self.mpc_controller.set_reference(x_ref_mpc, u_ref_mpc, warm_start_option=self.warm_start_option)
                u_opt, x_opt = self.mpc_controller.optimize(state_vi, return_x=True) # note MPC use state_vi
                # if use x_opt, use x_opt[1,:]
                cmd = u_opt[:4]
            else:
                # Use LQR
                # cmd = self.lqrcontroller.controller_thrusts(state, x_des)

                # Use MPC. if use x_opt, use x_opt[1,:]
                # interpolate x_ref and u_ref
                # x_ref_interpolated = np.zeros((self.mpc_controller.n_nodes + 1, self.n_state))
                # for i in range(7):
                #     x_ref_interpolated[:, i] = np.linspace(state_vi[i], x_des[i], self.mpc_controller.n_nodes + 1)
                # u_ref = np.ones((self.mpc_controller.n_nodes, 4)) * (self.mass*9.81/4)
                # self.mpc_controller.set_reference(x_ref_interpolated, u_ref, warm_start_option=self.warm_start_option)
                self.mpc_controller.set_reference(x_des)
                u_opt, x_opt = self.mpc_controller.optimize(state_vi, return_x=True) # note MPC use state_vi
                cmd = u_opt[:4]


            t1 = cmd[0]
            t2 = cmd[1]
            t3 = cmd[2]
            t4 = cmd[3]
            kt = 0.00026
            m1 = np.sign(t1)*np.sqrt(np.abs(t1)/kt)
            m2 = np.sign(t2)*np.sqrt(np.abs(t2)/kt)
            m3 = np.sign(t3)*np.sqrt(np.abs(t3)/kt)
            m4 = np.sign(t4)*np.sqrt(np.abs(t4)/kt)

            # if t1<0 or t2<0 or t3<0 or t4<0:
            #     rospy.loginfo("Negative thrust command detected.")

            # motors_vel = np.clip([m1, m2, m3, m4], 31, 155.043)# min 1 N, max 25 N
            # motors_vel = np.clip([m1, m2, m3, m4], 0.0, 165.043)# min 0 N, max > 25 N
            motors_vel = np.clip([m1, m2, m3, m4], -155.043, 155.043)# min 1 N, max 25 N

            self.__propellers[0].setVelocity(-motors_vel[0])
            self.__propellers[1].setVelocity(motors_vel[1])
            self.__propellers[2].setVelocity(motors_vel[2])
            self.__propellers[3].setVelocity(-motors_vel[3])
            # End of webots loop


    def handle_command(self, msg:String):
        command = msg.data.lower()
        if command == "takeoff" and (self.state == "waiting command" or self.state == "land"):
            self.takeoff()
        elif command == "hover" and (self.state == "takeoff" or self.state == "gotopoint" or "trajectory tracking"):
            pos = self.odom[0:3]
            if self.odom_frd:
                self.hover_pose[0:3] = pos # note frame
            else:
                self.hover_pose[0:3] = pos
            self.hover()
        elif command == "gotopoint" and (self.state == "hover" or self.state == "trajectory tracking"):
            self.gotopoint()
        elif command == "trajectory" and (self.state == "hover" or self.state == "gotopoint"):
            self.trajectory_tracking()
        elif command == "land" and (self.state != "waiting command"):
            self.land()

    def takeoff(self):
        self.state = 'takeoff'
        # Code to take off the drone
        rospy.loginfo('Taking off...')

        if not self.mission_finished:
            pass
        else:
            self.mission_finished = False

        # Transition to hover after takeoff
        # print(np.abs(-self.odom.position[2] - self.takeoff_height))

    def hover(self):
        
        self.state = 'hover'
        rospy.loginfo('Hovering...')
        # rospy.loginfo(f'Hovering...{self.hover_p}')
        # Send hover command or maintain altitude

    def gotopoint(self):
        self.goto_p[0:3] = np.array([self.target_position.pose.position.x, 
                                self.target_position.pose.position.y, 
                                self.target_position.pose.position.z])
        target_orientation = self.target_position.pose.orientation
        if target_orientation.w < 0.0: # Avoid quaternion flipping
            target_orientation.w *= -1
            target_orientation.z *= -1
            target_orientation.x *= -1
            target_orientation.y *= -1
        # target_orientation.w = np.sqrt(1 - target_orientation.z**2)
        self.goto_p[3:7] = np.array([target_orientation.w, target_orientation.x, target_orientation.y, target_orientation.z])
        self.state = 'gotopoint'
        rospy.loginfo('Going to point...')
        # Code to move to a specific point
        # Assume it calls self.hover() when point is reached
    
    def update_target_position(self, msg:PoseStamped):
        # Update the target position from the topic
        self.target_position = msg
        rospy.loginfo(f'Received new target position: {self.target_position.pose.position.x}, {self.target_position.pose.position.y}, {self.target_position.pose.position.z}')

    def trajectory_tracking(self):
        self.state = 'trajectory tracking'
        rospy.loginfo('Following trajectory...')
        # Code to follow trajectory
        # Assume it calls self.hover() when trajectory is complete

    def land(self):
        if self.odom_frd:
            self.landing_his = -self.odom[2]*self.landing_his
        else:
            self.landing_his = self.odom[2]*self.landing_his
        self.state = 'land'
        rospy.loginfo('Landing...')
        # Code to land the drone

    def waiting_command(self):
        self.waiting_pos = self.odom[0:3]
        self.state = 'waiting command'
        rospy.loginfo('Waiting for command...')

    def traj_callback(self, msg:ReferenceTrajectory):
        # Save reference name
        # self.ref_traj_name = msg.traj_name
        seq_len = msg.seq_len

        # Save reference trajectory, relative times and inputs
        x_ref_scvx = np.array(msg.trajectory).reshape(seq_len, -1)
        t_ref_scvx = np.array(msg.time)
        u_ref_scvx = np.array(msg.inputs).reshape(seq_len, -1)
        # self.quad_trajectory = np.zeros((len(self.t_ref), len(self.x)))
        # self.quad_controls = np.zeros((len(self.t_ref), 4))

        # self.w_control = np.zeros((len(self.t_ref), 3))

        dt = 1.0/self.control_hz
        self.t_ref = np.arange(0, (np.ceil((t_ref_scvx[-1])/dt)+1)*dt, dt)
        n_state = 13
        n_control = 4
        self.x_ref = np.zeros((self.t_ref.shape[0], n_state))
        self.u_ref = np.zeros((self.t_ref.shape[0], n_control))
        
        self.x_ref[:, 0] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 0])
        self.x_ref[:, 1] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 1])
        self.x_ref[:, 2] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 2])

        self.x_ref[:, 3] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 6])
        self.x_ref[:, 4] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 7])
        self.x_ref[:, 5] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 8])
        self.x_ref[:, 6] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 9])

        self.x_ref[:, 7] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 3])
        self.x_ref[:, 8] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 4])
        self.x_ref[:, 9] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 5])

        self.x_ref[:, 10] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 10])
        self.x_ref[:, 11] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 11])
        self.x_ref[:, 12] = np.interp(self.t_ref, t_ref_scvx, x_ref_scvx[:, 12])

        self.u_ref[:, 0] = np.interp(self.t_ref, t_ref_scvx, u_ref_scvx[:, 0])
        self.u_ref[:, 1] = np.interp(self.t_ref, t_ref_scvx, u_ref_scvx[:, 1])
        self.u_ref[:, 2] = np.interp(self.t_ref, t_ref_scvx, u_ref_scvx[:, 2])
        self.u_ref[:, 3] = np.interp(self.t_ref, t_ref_scvx, u_ref_scvx[:, 3])


        if self.state == "hover":
            self.state = "trajectory tracking"
        rospy.loginfo("New trajectory received.")
        # self.__node.get_logger().info("New trajectory received. Time duration: %.2f s" % self.t_ref[-1])


if __name__ == '__main__':
    MavicDriver()