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

    def img_coords2pose(self, cm, dir_vec, d_img, rot=None):
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
        #self.grasp_at_pose(pose_name)
        #self.deposit_obj(class_num)
        #self.deposit_obj_fake_ar(class_num % 4)

    def transform_head_rgbd_sensor_pixel_to_map(self, position, depth):
        pose_3d = self.robot.transform_pixel_to_pose(position)
        print(pose_3d)
        rgbd_pose = geometry_msgs.msg.PoseStamped()
        rgbd_pose.header.stamp = rospy.Time.now()
        rgbd_pose.header.frame_id = "head_rgbd_sensor_link"
        rgbd_pose.pose.position.x = pose_3d[0]
        rgbd_pose.pose.position.y = pose_3d[1]
        rgbd_pose.pose.position.z = pose_3d[2]
        trans = self.tfBuffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time())
        pose_transformed = tf2_geometry_msgs.do_transform_pose(rgbd_pose, trans)
        rgbd_pub = rospy.Publisher('desired_grasp_center_rgbd_frame', geometry_msgs.msg.PoseStamped, queue_size=10)
        map_pub = rospy.Publisher('desired_grasp_center_map_frame', geometry_msgs.msg.PoseStamped, queue_size=10)
        while True:
            rgbd_pub.publish(rgbd_pose)
            map_pub.publish(pose_transformed)
        return pose_transformed

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
        # rotate angle by 180 degrees if dexnet gives angle out of bound for hsr
        if grasp_angle_dexnet < -1.92:
            grasp_angle_hsr = grasp_angle_dexnet + 3.14
        else:
            grasp_angle_hsr = grasp_angle_dexnet
        return grasp_angle_hsr

    def adjust_grasp_center(self, desired_grasp_center, actual_grasp_center):
        difference_x = desired_grasp_center.pose.position.x - actual_grasp_center.pose.position.x
        difference_y = desired_grasp_center.pose.position.y - actual_grasp_center.pose.position.y
        print('difference between them')
        print([difference_x, difference_y])
        self.robot.omni_base.go_rel(difference_x, difference_y, 0.0, 0)
        self.robot.whole_body.move_to_joint_positions({'head_pan_joint': 0.24998440577459347,
                                                        'head_tilt_joint': -1.3270548266651048})

    def go_to_start_pose(self, grasp_angle_hsr):
        self.robot.open_gripper()
        self.robot.whole_body.move_to_joint_positions({'arm_roll_joint': 0.0,
                                        'arm_lift_joint': 0.1,
                                        'arm_flex_joint': -1.83,
                                        'wrist_flex_joint': -1.31,
                                        'wrist_roll_joint': grasp_angle_hsr})

    def show_grasp_in_rviz(self, grasp_center, depth_m, grasp_angle_dexnet):
        grasp_angle_hsr = self.transform_dexnet_angle(grasp_angle_dexnet)
        desired_grasp_center = self.transform_head_rgbd_sensor_pixel_to_map(grasp_center, depth_m)
        trans_pub = rospy.Publisher('desired_grasp_center', geometry_msgs.msg.PoseStamped, queue_size=10)
        #while True:
        #    trans_pub.publish(desired_grasp_center)

    def execute_2DOF_grasp(self, grasp_center, depth_m, grasp_angle_dexnet):
        grasp_angle_hsr = self.transform_dexnet_angle(grasp_angle_dexnet)
        self.go_to_start_pose(grasp_angle_hsr)
        actual_grasp_center = self.compute_actual_grasp_center()
        desired_grasp_center = self.transform_head_rgbd_sensor_pixel_to_map(grasp_center, depth_m)
        print('desired')
        print(desired_grasp_center.pose.position)
        print('actual')
        print(actual_grasp_center.pose.position)
        while desired_grasp_center.pose.position.x - actual_grasp_center.pose.position.x >= 0.01 or desired_grasp_center.pose.position.y - actual_grasp_center.pose.position.y >= 0.01:
            self.adjust_grasp_center(desired_grasp_center, actual_grasp_center)
            actual_grasp_center = self.compute_actual_grasp_center()
        if depth_m > 0.91:
            z = 0
        else:
            z = 0.91 - depth_m
        self.robot.whole_body.move_to_joint_positions({'arm_lift_joint': z})
        #time.sleep(1)
        #self.robot.close_gripper()
        #time.sleep(1)
        #self.robot.whole_body.move_to_joint_positions({'arm_lift_joint': z + 0.2})


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
