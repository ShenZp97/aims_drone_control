#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
import os
from nav_msgs.msg import Odometry
import numpy as np
import yaml

from std_msgs.msg import String  # For sending commands 
from geometry_msgs.msg import Point, PoseStamped  # For positions
from aims_msgs.msg import ReferenceTrajectory
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandLong, CommandLongRequest

from DroneLQR import LQRController
from mpc_drone_optimizer import QuadrotorParam
from mpc_drone import Quad3DMPC
from time import time

import os 


class PX4Driver:
    def __init__(self):
        

        #-----FSM
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path+'/config/manta36_exp_bi.yaml','r') as f:
            config = yaml.safe_load(f)
        self.lqrcontroller = LQRController(config)
        drone_config = QuadrotorParam(config)
        self.mass = config['mass']
        self.warm_start_option = config['warm_start_option']
        self.max_total_t = drone_config.max_total_t
        self.min_total_t = drone_config.min_total_t
        self.mpc_controller = Quad3DMPC(drone_config,
                                        r_cost=config['r_cost'],
                                        q_cost=config['q_cost'])

        self.state = "waiting command"
        self.tracking = False
        self.reach_tolerance = 0.3
        
        self.odom = np.zeros(13)
        self.takeoff_height = config['takeoff_h']
        self.control_hz = config["control_hz"] # control frequency
        self.odom_frd = False # Is the frame of odom a frd frame?
        self.cmd_frd = False # Is the frame of cmd a frd frame?
        self.hover_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.goto_p =  np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.landing_his = np.ones(8*self.control_hz) # landing history for ground detection
        self.landing_dis = 1.0 # point distance for landing phase
        self.mission_finished = False
        self.waiting_pos = np.array([0.0, 0.0, 0.0])
        # self.traj = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        self.n_state = 13
        self.n_control = 4

        self.start_time = time()
        self.px4_pose_time = time()

        self.tracking_time = 0.0

        #ROS
        rospy.init_node('px4_control', anonymous=False)

        self.command_subscriber = rospy.Subscriber('drone_command', String, self.handle_command)
        self.goto_p_subscriber = rospy.Subscriber('goto_position', PoseStamped, self.update_target_position)
        self.traj_subscriber = rospy.Subscriber('trajectory', ReferenceTrajectory, self.traj_callback)

        # self.local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        # self.attitude_pub = rospy.Publisher('/mavros/setpoint_attitude/attitude', PoseStamped, queue_size=1)
        # Px4 offboard control
        odom_topic = "odom"
        self.odom_sub =rospy.Subscriber(
                odom_topic, Odometry, self.odometry_callback, queue_size=1, tcp_nodelay=True)
        
        control_topic = "mavros/setpoint_raw/attitude"
        self.rate = rospy.Rate(self.control_hz)
        self.control_pub = rospy.Publisher(control_topic, AttitudeTarget, queue_size=1, tcp_nodelay=True)
        self.cmd_px4 = AttitudeTarget()

        px4_pose_topic = "mavros/vision_pose/pose" # give vision pose to px4
        self.vision_pose_msg = PoseStamped()
        self.vision_pub = rospy.Publisher(px4_pose_topic, PoseStamped, queue_size=1)# give vision pose to px4

        rospy.wait_for_service("mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        rospy.wait_for_service("mavros/cmd/command")
        self.command_client = rospy.ServiceProxy("mavros/cmd/command", CommandLong)
        
        self.px4_state = None
        self.state_sub = rospy.Subscriber("mavros/state", State, callback = self.state_cb)
        
        rospy.loginfo('Controller initialized')

        rospy.spin()

    def state_cb(self, msg:State):
        self.px4_state = msg
    
    def odometry_callback(self, msg:Odometry):

        self.odom[0] = msg.pose.pose.position.x
        self.odom[1] = msg.pose.pose.position.y
        self.odom[2] = msg.pose.pose.position.z
        self.odom[3] = msg.pose.pose.orientation.w
        self.odom[4] = msg.pose.pose.orientation.x
        self.odom[5] = msg.pose.pose.orientation.y
        self.odom[6] = msg.pose.pose.orientation.z
        self.odom[7] = msg.twist.twist.linear.x
        self.odom[8] = msg.twist.twist.linear.y
        self.odom[9] = msg.twist.twist.linear.z
        self.odom[10] = msg.twist.twist.angular.x
        self.odom[11] = msg.twist.twist.angular.y
        self.odom[12] = msg.twist.twist.angular.z

        # Give Pose info to PX4
        elapsed_pose_time = time() - self.px4_pose_time
        if elapsed_pose_time > 1.0/50.0: # 50 Hz
            self.px4_pose_time = time()
            self.vision_pose_msg.header.frame_id = "world"
            self.vision_pose_msg.header.stamp = rospy.Time.now()
            self.vision_pose_msg.pose.position.x = msg.pose.pose.position.x
            self.vision_pose_msg.pose.position.y = msg.pose.pose.position.y
            self.vision_pose_msg.pose.position.z = msg.pose.pose.position.z
            self.vision_pose_msg.pose.orientation.w = msg.pose.pose.orientation.w
            self.vision_pose_msg.pose.orientation.x = msg.pose.pose.orientation.x
            self.vision_pose_msg.pose.orientation.y = msg.pose.pose.orientation.y
            self.vision_pose_msg.pose.orientation.z = msg.pose.pose.orientation.z
            self.vision_pub.publish(self.vision_pose_msg)

        elapsed_time = time() - self.start_time
        if elapsed_time > 1.0/self.control_hz:
            self.start_time = time()
            self.control_loop()

    def control_loop(self):
        p_flu = self.odom[0:3]
        q_flu = self.odom[3:7]
        v_flu = self.odom[7:10]
        w_flu = self.odom[10:13]

        # FRD to FLU
        if self.odom_frd:
            rospy.logwarn("Mavros should use FLU frame")
            # p_flu = np.array([pos.x, -pos.y, -pos.z])
            # q_flu = np.array([q.w, q.x, -q.y, -q.z])
            # Q = self.lqrcontroller.qtoQ(q_flu) # body to inertial
            # v_flu = np.array([v.x, -v.y, -v.z])
            # v_b = Q.T @ v_flu
            # w_flu = np.array([w.x, -w.y, -w.z])
        # else: # body velocity
            # Q = self.lqrcontroller.qtoQ(q_flu) # body to inertial
            # v_b = Q.T @ v_flu

        # state = np.hstack((p_flu, q_flu, v_b, w_flu)) # converted to FLU
        state_vi = np.hstack((p_flu, q_flu, v_flu, w_flu)) # converted to FLU. Not body velocity
        
        if self.state == "trajectory tracking":
            if not self.tracking:
                if np.linalg.norm(self.x_ref[0, 0:3] - p_flu) > self.reach_tolerance:
                    x_des = np.hstack((self.x_ref[0, 0:3], np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)))
                    rospy.loginfo_once("Raising drone to the start point of the trajectory...")
                else:
                    self.tracking_time = rospy.Time.now().to_sec()
                    self.tracking = True

            if self.tracking:
                if np.linalg.norm(self.x_ref[-1, 0:3]-p_flu) <= self.reach_tolerance and ((rospy.Time.now().to_sec() - self.tracking_time)>=1.0): # reached end of trajectory
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
                    t_ref_now = rospy.Time.now().to_sec() - self.tracking_time

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
            if (np.max(self.landing_his)-np.min(self.landing_his))<=0.2: # reached floor
                # disarm
                # while self.px4_state.armed:
                #     arm_cmd = CommandBoolRequest()
                #     arm_cmd.value = False
                #     self.arming_client.call(arm_cmd)

                # force to shutdown
                shutdown_cmd = CommandLongRequest()
                shutdown_cmd.broadcast = False
                shutdown_cmd.confirmation = 0
                shutdown_cmd.command = 400
                shutdown_cmd.param1 = 0
                shutdown_cmd.param2 = 21196
                shutdown_cmd.param3 = 0
                shutdown_cmd.param4 = 0
                shutdown_cmd.param5 = 0
                shutdown_cmd.param6 = 0
                shutdown_cmd.param7 = 0

                if self.command_client.call(shutdown_cmd).success == True:
                    rospy.loginfo('Shutdown command sent.')

                self.waiting_command()
                self.mission_finished = True
                        
            else:
                x_des = np.hstack((p_land, np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)))
            
        if self.state=="hover":
            x_des = np.hstack((self.hover_pose, np.zeros(3), np.zeros(3)))

        # compute control
        if self.state == "waiting command": # do nothing
            rates = np.zeros(3)
            t_total = 0.0
        elif self.state == "trajectory tracking" and self.tracking: # use MPC
            # rospy.loginfo("MPC tracking.")
            self.mpc_controller.set_reference(x_ref_mpc, u_ref_mpc, warm_start_option=self.warm_start_option)
            u_opt, x_opt = self.mpc_controller.optimize(state_vi, return_x=True) # note MPC use state_vi
            # if use x_opt, use x_opt[1,:]
            # t_total = sum(u_opt[:4])/self.max_total_t
            
            thrust_all = sum(u_opt[:4])
            # mapping positive thrust to 0.5 to 1.0, negative thrust to 0.0 to 0.5
            if thrust_all >= 0:
                t_total = 0.5 + 0.5*(thrust_all/self.max_total_t)
            else:
                t_total = 0.5 - 0.5*(thrust_all/self.min_total_t)

            rates = x_opt[1, 10:13]
        else:
            # Use LQR
            # rates, t_total = self.lqrcontroller.controller_px4rates(state, x_des)

            # Use MPC. if use x_opt, use x_opt[1,:]
            self.mpc_controller.set_reference(x_des)
            u_opt, x_opt = self.mpc_controller.optimize(state_vi, return_x=True) # note MPC use state_vi
            # t_total = sum(u_opt[:4])/self.max_total_t

            thrust_all = sum(u_opt[:4])
            # mapping positive thrust to 0.5 to 1.0, negative thrust to 0.0 to 0.5
            if thrust_all >= 0:
                t_total = 0.5 + 0.5*(thrust_all/self.max_total_t)
            else:
                t_total = 0.5 - 0.5*(thrust_all/self.min_total_t)

            rates = x_opt[1, 10:13]

        # Publish control
        self.cmd_px4.header.stamp = rospy.Time.now()
        self.cmd_px4.type_mask = 128 # ignore attitude
        self.cmd_px4.body_rate.x = rates[0]
        self.cmd_px4.body_rate.y = rates[1]
        self.cmd_px4.body_rate.z = rates[2]
        self.cmd_px4.thrust = t_total
        self.control_pub.publish(self.cmd_px4)


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
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'

        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        while self.px4_state.mode != 'OFFBOARD':
            if self.set_mode_client.call(offb_set_mode) == True:
                rospy.loginfo('Offboard mode enabled.')
                break

        while not self.px4_state.armed:
            if self.arming_client.call(arm_cmd) == True:
                rospy.loginfo('Vehicle armed.')
                break

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

        # target_orientation = self.target_position.pose.orientation
        # if target_orientation.w < 0.0: # Avoid quaternion flipping
        #     target_orientation.w *= -1
        #     target_orientation.z *= -1
        # target_orientation.w = np.sqrt(1 - target_orientation.z**2)
        # self.goto_p[3:7] = np.array([target_orientation.w, 0.0, 0.0, target_orientation.z])

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
    PX4Driver()