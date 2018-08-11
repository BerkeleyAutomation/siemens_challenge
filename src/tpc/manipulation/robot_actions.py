import sys
import IPython
import tpc.config.config_tpc as cfg
import numpy as np
import time

class Robot_Actions():
    def __init__(self, robot):
        self.robot = robot

    def safe_wait(self):
        #making sure the robot is finished moving
        time.sleep(3)

    def go_to_start_pose(self):
        self.robot.body_start_pose()
        self.robot.close_gripper()
        self.safe_wait()

    def head_start_pose(self):
        self.robot.head_start_pose()
        self.safe_wait()

    def go_to_start_position(self, offsets=None):
        self.robot.position_start_pose(offsets=offsets)
        self.safe_wait()

    def get_position(self):
        return self.robot.get_position()

    def move_to_start(self, z=True, y=True, x=True):
        curr_pos = self.get_position()
        start_pos = self.get_start_position()
        print(curr_pos, start_pos)

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
        return self.robot.get_start_position()

    def set_start_position(self):
        self.robot.set_start_position()

    def img_coords2pose(self, cm, dir_vec, d_img, rot=None):
        z = self.robot.get_depth(cm, d_img)
        if rot is None:
            rot = self.robot.get_rot(dir_vec)

        pose_name = self.robot.create_grasp_pose(cm[1], cm[0], z, rot)
        time.sleep(2)
        return pose_name

    def grasp_at_pose(self, pose_name):
        self.robot.open_gripper()
        self.robot.move_to_pose(pose_name, 0.1)
        self.robot.move_to_pose(pose_name, 0.022)
        self.robot.close_gripper()
        self.robot.move_to_pose(pose_name, 0.1)
        # self.robot.move_to_pose(pose_name, 0.2)
        # self.robot.move_to_pose(pose_name, 0.1)

    def deposit_obj(self, class_num):
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
            while not found and i < 10:
                found = self.robot.find_ar(class_num + 12) #AR numbers from 8 to 11
                if not found:
                    print(i)
                    curr_tilt = -1 + (i * 1.0)/2.0 #ranges from -1 to 1
                    self.robot.pan_head(curr_tilt)
                    i += 1
            if not found:
                print("Could not find AR marker- depositing object in default position.")
                self.go_to_start_position(offsets=[-0.5, 0, 0])

        self.move_base(x=0.1)
        self.robot.open_gripper()
        self.robot.close_gripper()
        self.move_base(x=-0.1)

    def deposit_in_cubby(self):
        self.robot.move_in_cubby()
        self.robot.open_gripper()
        self.robot.close_gripper()
        self.robot.body_neutral_pose()
        self.robot.body_start_pose()

    def move_base(self, x=0, y=0, z=0):
        self.robot.move_base(x=x, y=y, z=z)

    def execute_grasp(self, cm, dir_vec, d_img, class_num):
        pose_name = self.img_coords2pose(cm, dir_vec, d_img)
        self.grasp_at_pose(pose_name)
        # self.deposit_obj(class_num)

    def spread_singulate(self, cm, dir_vec, d_img):
        pose_name = self.img_coords2pose(cm, dir_vec, d_img)
        self.robot.move_to_pose(pose_name, 0.06, y_offset=0.09)
        self.robot.move_to_pose(pose_name, 0.022, y_offset=-0.05)
        self.robot.move_to_pose(pose_name, 0.022, x_offset=0.1)
        self.robot.move_to_pose(pose_name, 0.1)
        # self.deposit_obj(class_num)

    def execute_singulate(self, waypoints, rot, d_img):
        self.robot.close_gripper()

        pose_names = [self.img_coords2pose(waypoint, None, d_img, rot=rot) for waypoint in waypoints]

        self.robot.move_to_pose(pose_names[0], 0.05)

        for pose_name in pose_names:
            self.robot.move_to_pose(pose_name, 0.03)

        self.robot.move_to_pose(pose_names[-1], 0.05)
