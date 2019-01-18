import sys
import IPython
import tpc.config.config_tpc as cfg
import numpy as np
import time
import math
import rospy
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs
import thread
import os
import tf.transformations as transformations

class Robot_Actions():
    """
    high level declutering task-specific interface that interacts 
    with the HSR_CORE Robot_Interface:
    https://github.com/BerkeleyAutomation/HSR_COR

    '''
    Attributes
    ----------
    robot : Robot_Interface
        low level interface that interacts with primitive HSR kinematics:
        http://hsr.io/
    """

    def __init__(self, robot):
        self.robot = robot
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def safe_wait(self):
        """
        used to make sure robot is
        finished moving

        """
        # time.sleep(3)
        time.sleep(0.1)

    def go_to_start_pose(self):
        """
        return to the starting body and
        gripper positions

        """
        self.robot.body_start_pose()
        self.robot.close_gripper()
        self.safe_wait()

    def head_start_pose(self):
        """
        return to the starting head
        pose (facing forward)

        """
        self.robot.head_start_pose()
        self.safe_wait()
        self.safe_wait()

    def go_to_start_position(self, offsets=None):
        """
        return to the starting base
        position in single motion

        '''
        Parameters
        ----------
        offsets : list
            list of x, y, z offsets
            from the starting postion

        """
        self.robot.position_start_pose(offsets=offsets)
        self.safe_wait()

    def get_position(self):
        """
        get the current base position
        
        '''
        Returns
        -------
        []
            a list of x, y, z coordinates

        """
        return self.robot.get_position()

    def move_to_start(self, z=True, y=True, x=True):
        """
        move to the starting base position
        in separated x, y, and z motions

        '''
        Parameters
        ----------
        x, y, z : float
            positional offsets to move to
            relative from starting position
        """
        curr_pos = self.get_position()
        start_pos = self.get_start_position()
        # print(curr_pos, start_pos)

        y_dist = abs(curr_pos[1] - start_pos[1])
        x_dist = abs(curr_pos[0] - start_pos[0])
        z_dist = abs(curr_pos[2] - start_pos[2])

        if z:
            if curr_pos[2] > start_pos[2]:
                self.move_base(z=-z_dist)
            else:
                self.move_base(z=z_dist)
        if y:
            if curr_pos[1] > start_pos[1]:
                self.move_base(y=-y_dist)
            else:
                self.move_base(y=y_dist)
        if x:
            if curr_pos[0] > start_pos[0]:
                self.move_base(x=-x_dist)
            else:
                self.move_base(x=x_dist)

    def get_start_position(self):
        """
        get the starting base position
        
        '''
        Return
        ------
        The starting base position

        """
        return self.robot.get_start_position()

    def set_start_position(self):
        """
        set the starting base position

        """
        self.robot.set_start_position()

    def img_coords2pose(self, cm, dir_vec, d_img, rot=None, depth=None):
        """
        convert image coordinates to a
        grasp pose

        '''
        Parameters
        ----------
        cm : []
            x, y pixel coordinates of center of mass
        dir_vec : []
            vector of direction to singulate in
        d_img : cv2.img
            depth image from robot
        rot : boolean
            True if singulate, None otherwise

        Returns
        -------
        name of the generated grasp pose

        """
        if depth is not None:
            z = depth
        else:
            z = self.robot.get_depth(cm, d_img)
        if rot is None:
            rot = self.robot.get_rot(dir_vec)
        pose_name = self.robot.create_grasp_pose(cm[1], cm[0], z, rot)
        # time.sleep(2)
        time.sleep(0.1)
        return pose_name

    def grasp_at_pose(self, pose_name):
        """
        grasp object at the given grasp pose name

        '''
        Parameters
        ----------
        pose_name : String
            name of the pose to grasp at

        """
        self.robot.open_gripper()
        # import ipdb;ipdb.set_trace()
        self.robot.move_to_pose(pose_name, 0.1)
        self.robot.move_to_pose(pose_name, 0.02)
        # self.robot.move_to_pose(pose_name, 0.019)
        # time.sleep(1)
        # self.robot.move_to_pose(pose_name, 0.015)
        # self.robot.move_to_pose(pose_name, 0.01)
        self.robot.close_gripper()
        # self.robot.move_to_pose(pose_name, 0.2)
        # self.robot.move_to_pose(pose_name, 0.2)
        self.robot.move_to_pose(pose_name, 0.1)

    def deposit_obj(self, class_num):
        """
        deposit the object in a bin

        '''
        Parameters
        ----------
        class_num : int
            class label number to determine
            which bin to deposit in

        """
        if class_num is None:
            #go to a temporary pose for the bins
            self.go_to_start_position(offsets=[-0.5, 0, 0])
        else:
            # print("Class is " + cfg.labels[class_num])
            print("Class is " + str(class_num))
            # self.robot.pan_head()
            # self.robot.tilt_head(tilt=-0.4)
            found = False
            i = 0
            self.robot.pan_head(-1.2)
            while not found and i < 10:
                found = self.robot.find_ar(class_num + 8) #AR numbers from 8 to 11
                if not found:
                    print(i)
                    # if i == 0:
                    curr_tilt = -1.0 + (i * 0.1)/2.0 #ranges from -1 to 1
                    self.robot.tilt_head(curr_tilt)
                    # self.robot.pan_head(curr_tilt)

                    i += 1
            if not found:
                print("Could not find AR marker- depositing object in default position.")
                self.go_to_start_position(offsets=[-0.5, 0, 0])

        self.move_base(x=0.1)
        self.robot.open_gripper()
        self.robot.close_gripper()
        self.move_base(x=-0.1)
        self.safe_wait()

    def deposit_obj_fake_ar(self, class_num):
        # import ipdb; ipdb.set_trace()
        if class_num is None:
            #go to a temporary pose for the bins
            self.go_to_start_position(offsets=[-0.5, 0, 0])
        else:
            # print("Class is " + cfg.labels[class_num])
            print("Class is " + str(class_num))
            # self.robot.pan_head()
            # self.robot.tilt_head(tilt=-0.4)

            marker_name = 'fake_ar'+str(class_num)
            self.robot.move_to_fake_ar(ar_name=marker_name)

            # found = False
            # i = 0
            # self.robot.pan_head(-1.2)
            # while not found and i < 10:
            #     found = self.robot.find_ar(class_num + 8) #AR numbers from 8 to 11
            #     if not found:
            #         print(i)
            #         # if i == 0:
            #         curr_tilt = -1.0 + (i * 0.1)/2.0 #ranges from -1 to 1
            #         self.robot.tilt_head(curr_tilt)
            #         # self.robot.pan_head(curr_tilt)
            #
            #         i += 1
            # if not found:
            #     print("Could not find AR marker- depositing object in default position.")
            #     self.go_to_start_position(offsets=[-0.5, 0, 0])

        # self.move_base(x=0.1)
        self.robot.open_gripper()
        self.robot.close_gripper()
        # self.move_base(x=-0.1)
        self.safe_wait()

    def deposit_in_cubby(self, x_pos=0.0, z_pos=0.0, label=None):
        """
        deposit the object in a cubby

        """
        self.robot.move_in_cubby(x_pos=x_pos, z_pos=z_pos)
        if label == 3:
            self.robot.move_base(z=0.233)
        self.robot.open_gripper()
        self.robot.close_gripper()
        if label == 3:
            self.robot.move_base(z=-0.233)
        self.robot.body_neutral_pose()
        self.robot.body_start_pose()

    def move_base(self, x=0, y=0, z=0):
        """
        move base of robot to desired
        x, y, z coordinates

        '''
        Parameters
        ----------
        x, y, z : float
            coordinates to move to

        """
        self.robot.move_base(x=x, y=y, z=z)

    def execute_grasp(self, cm, dir_vec, d_img, class_num, lin_weight=None):
        """
        execute a grasp

        '''
        Paramters
        ---------
        cm : []
            x, y, z, pixel coordinates of center
            of mass
        dir_vec : []
            vector of direction to push
        d_img : cv2.img
            depth image from robot
        class_num : int
            class label number to determine
            which bin to deposit in

        """
        if lin_weight is not None:
            self.robot.linear_weight = lin_weight
        pose_name = self.img_coords2pose(cm, dir_vec, d_img)
        print('Grasp?')
        x = raw_input()
        if x == 'exit':
            return
        self.grasp_at_pose(pose_name)
        #self.deposit_obj(class_num)
        #self.deposit_obj_fake_ar(class_num % 4)

    def compute_actual_grasp_center(self):
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "hand_palm_link"
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        pose.pose.position.z = 0
        trans = self.tfBuffer.lookup_transform('map', 'hand_palm_link', rospy.Time())
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def transform_dexnet_angle(self, grasp_angle_dexnet):
        # rgbd camera is rotated by 0.32 radian, need to compensate that in angle
        grasp_angle_dexnet -= 0.32
        # rotate angle by 180 degrees if dexnet gives angle out of bound for hsr
        if grasp_angle_dexnet < -1.92:
            grasp_angle_hsr = grasp_angle_dexnet + np.pi
        else:
            grasp_angle_hsr = grasp_angle_dexnet
        return grasp_angle_hsr

    def adjust_grasp_center(self, desired_grasp_center, actual_grasp_center):
        difference_x = desired_grasp_center.pose.position.x - actual_grasp_center.pose.position.x
        difference_y = desired_grasp_center.pose.position.y - actual_grasp_center.pose.position.y
        base_position_map_frame = self.robot.omni_base.get_pose()
        quaternion_matrix = transformations.quaternion_matrix(base_position_map_frame.ori)
        euler_angles = transformations.euler_from_matrix(quaternion_matrix)
        self.robot.omni_base.go_abs(base_position_map_frame.pos.x + difference_x, base_position_map_frame.pos.y + difference_y, euler_angles[2], 0)
        base_position_map_frame = self.robot.omni_base.get_pose()

    def go_to_start_pose(self):
        self.robot.whole_body.move_to_joint_positions({'arm_flex_joint': -0.005953039901891888,
                                        'arm_lift_joint': 3.5673664703075522e-06,
                                        'arm_roll_joint': -1.6400026753088877,
                                        'head_pan_joint': 0.24998440577459347,
                                        'head_tilt_joint': -1.3270548266651048,
                                        'wrist_flex_joint': -1.570003402348724,
                                        'wrist_roll_joint': 0})

    def go_to_grasp_pose(self, grasp_angle_hsr):
        self.robot.open_gripper()
        self.robot.whole_body.move_to_joint_positions({'arm_roll_joint': -0.04,
                                        'arm_lift_joint': 0.15,
                                        'arm_flex_joint': -1.91,
                                        'wrist_flex_joint': -1.18,
                                        'wrist_roll_joint': grasp_angle_hsr})
        time.sleep(1)

    def get_desired_grasp_in_map_frame(self):
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "grasp_i_0"
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        pose.pose.position.z = 0
        trans = self.tfBuffer.lookup_transform('map', 'grasp_i_0', rospy.Time())
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        pub = rospy.Publisher('new_desired_grasp', geometry_msgs.msg.PoseStamped, queue_size=10)
        return pose_transformed

    def compute_z_value(self, desired_grasp_center):
        # z value is the distance between the floor and the middle of the gripper
        # the new depth sampling from gqcnn outputs the desired grasp on the object surface, i.e. on top of the object. Thus, we lower it by offset_from_object_surface
        offset_from_object_surface = 0.02
        floor_z_value_in_map_frame = 0.005
        gripper_height = 0.008
        z = desired_grasp_center.pose.position.z
        z -= offset_from_object_surface
        z -= floor_z_value_in_map_frame
        z -= gripper_height
        if z < 0:
            z = 0
        return z

    def adjust_z_based_on_grasp_width(self, z, grasp_width):
        if grasp_width <= 0.02:
            adjustment = 0.016 + 0.004 * (0.02 - grasp_width) / 0.02
        elif 0.02 < grasp_width <= 0.09:
            adjustment = 0.009 + 0.005 * (0.09 - grasp_width) / 0.07
        elif 0.09 < grasp_width <= 0.11:
            adjustment = 0.002 + 0.007 * (0.11 - grasp_width) / 0.02
        else:
            adjustment = 0.002
        z += adjustment
        return z

    def execute_2DOF_grasp(self, grasp_center, grasp_depth_m, grasp_angle_dexnet, grasp_width, d_img):
        grasp_angle_hsr = self.transform_dexnet_angle(grasp_angle_dexnet)
        # use dummy direction because this function needs one as argument
        dir_vec = [0, 1]
        # img_coords2pose exchanges x and y of grasp center, thus having to give them exchanged
        self.img_coords2pose([grasp_center[1], grasp_center[0]], dir_vec, d_img, depth=grasp_depth_m*1000)
        desired_grasp_center = self.get_desired_grasp_in_map_frame()
        exit_var = raw_input()
        if exit_var == 'exit':
            return
        self.go_to_grasp_pose(grasp_angle_hsr)
        time.sleep(1)
        actual_grasp_center = self.compute_actual_grasp_center()
        self.adjust_grasp_center(desired_grasp_center, actual_grasp_center)
        z = self.compute_z_value(desired_grasp_center)
        self.robot.whole_body.move_to_joint_positions({'arm_lift_joint': z})
        z = self.adjust_z_based_on_grasp_width(z, grasp_width)
        #time.sleep(5)
        self.robot.whole_body.move_to_joint_positions({'arm_lift_joint': z})
        #time.sleep(5)
        self.robot.close_gripper()
        #time.sleep(5)
        self.robot.whole_body.move_to_joint_positions({'arm_lift_joint': z + 0.2})


    def l_singulate(self, cm, dir_vec, d_img):
        """
        perform an L-shaped singulation
        (push forward then to the side)
        
        '''
        Parameters
        ----------
        cm : []
            x, y, z, pixel coordinates of center
            of mass
        dir_vec : []
            vector of direction to push
        d_img : cv2.img
            depth image from robot

        """

        pose_name = self.img_coords2pose(cm, dir_vec, d_img)
        self.robot.move_to_pose(pose_name, 0.06, y_offset=-0.1)
        self.robot.move_to_pose(pose_name, 0.021, y_offset=-0.1)
        self.robot.move_to_pose(pose_name, 0.021, y_offset=0.04)
        self.robot.move_to_pose(pose_name, 0.021, x_offset=0.05)
        self.robot.move_to_pose(pose_name, 0.1)
        # self.deposit_obj(class_num)

    def execute_singulate(self, waypoints, rot, d_img):
        """
        perform a singulation determined by
        Singulator class
        
        '''
        Parameters
        ----------
        waypoints : np.array
            list of 1x2 vectors representing the waypoints 
            of the singulation
        float : []
            rotation of gripper for singulation
        d_img : cv2.img
            depth image from robot

        """
        self.robot.close_gripper()

        pose_names = [self.img_coords2pose(waypoint, None, d_img, rot=rot) for waypoint in waypoints]

        self.robot.move_to_pose(pose_names[0], 0.05)

        for pose_name in pose_names:
            self.robot.move_to_pose(pose_name, 0.04)

        self.robot.move_to_pose(pose_names[-1], 0.04)
        self.robot.move_to_pose(pose_names[-1], 0.15)
