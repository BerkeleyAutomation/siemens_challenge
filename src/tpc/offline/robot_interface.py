import cv2
import numpy as np

class Robot_Interface(object):
    def __init__(self):
        pass

    def body_start_pose(self):
        pass

    def head_start_pose(self):
        pass

    def position_start_pose(self, offsets=None):
        pass

    def get_img_data(self):
        sample_img_path = "data/example_images/frame_40_0.png"
        sample_img = cv2.imread(sample_img_path)
        sample_depth = np.zeros(sample_img.shape[:2])
        return sample_img, sample_depth

    def get_depth(self, point, d_img):
        return 0

    def get_rot(self, direction):
        return 0

    def create_grasp_pose(x, y, z, rot):
        return ""

    def open_gripper():
        pass

    def close_gripper():
        pass

    def move_to_pose(pose_name, z_offset):
        pass

    def find_ar(self, ar_number):
        return True

    def pan_head(self, tilt):
        pass
