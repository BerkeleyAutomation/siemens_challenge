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
        time.sleep(2)


    def go_to_start_pose(self):
        self.robot.body_start_pose()
        self.safe_wait()
        self.robot.head_start_pose()
        self.safe_wait()

    def go_to_start_position(self, offsets=None):
        self.robot.position_start_pose(offsets=offsets)
        self.safe_wait()

    def img_coords2pose(self, cm, dir_vec, d_img, rot=None):
        z = self.robot.get_depth(cm, d_img)
        if rot is None:
            rot = self.robot.get_rot(dir_vec)

        pose_name = self.robot.create_grasp_pose(cm[1], cm[0], z, rot)
        time.sleep(1)
        return pose_name

    def grasp_at_pose(self, pose_name):
        self.robot.open_gripper()
        self.robot.move_to_pose(pose_name, 0.15 + cfg.GRIPPER_LENGTH)
        # IPython.embed()
        self.safe_wait()
        self.robot.move_to_pose(pose_name, 0 + cfg.GRIPPER_LENGTH)
        self.robot.close_gripper()
        self.robot.move_to_pose(pose_name, 0.3)

    def deposit_obj(self, class_num):
        if class_num is None:
            #go to a temporary pose for the bins
            self.go_to_start_position(offsets=[-0.5, 0, 0])
        else:
            # IPython.embed()
            print("Class is " + cfg.labels[class_num])
            # print("Class is " + str(class_num))
            self.go_to_start_position()
            found = False
            i = 0
            while not found and i < 10:
                found = self.robot.find_ar(class_num + 8) #AR numbers from 8 to 11
                if not found:
                    print(i)
                    curr_tilt = -1 + (i * 1.0)/5.0 #ranges from -1 to 1
                    self.robot.pan_head(curr_tilt)
                    i += 1
            if not found:
                print("Could not find AR marker- depositing object in default position.")
                self.go_to_start_position(offsets=[-0.5, 0, 0])

        self.robot.open_gripper()

    def execute_grasp(self, cm, dir_vec, d_img, class_num):
        pose_name = self.img_coords2pose(cm, dir_vec, d_img)
        self.grasp_at_pose(pose_name)
        self.deposit_obj(class_num)

    def execute_singulate(self, waypoints, rot, d_img):
        self.robot.close_gripper()

        pose_names = [self.img_coords2pose(waypoint, None, d_img, rot=rot) for waypoint in waypoints]

        self.robot.move_to_pose(pose_names[0], 0.05)

        for pose_name in pose_names:
            self.robot.move_to_pose(pose_name, 0.03)

        self.robot.move_to_pose(pose_names[-1], 0.05)
