import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
import rospy
from copy import deepcopy
import sys
import thread
import tf

import importlib
import matplotlib.pyplot as plt

import hsrb_interface # DELETE THIS LATER


from hsrb_interface import geometry
from tpc.perception.cluster_registration import run_connected_components, display_grasps, class_num_to_name, \
    grasps_within_pile, hsv_classify
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, \
    find_isolated_objects_by_distance
from tpc.manipulation.robot_actions import Robot_Actions
import tpc.config.config_tpc as cfg
import tensorflow

from tpc.detection.detector import Detector


from hsr_core.hsr_robot_interface import Robot_Interface

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')


class DeclutterDemo():
    """
    This class is for use with the robot
    Pipeline it tests is labeling all objects in an image using web interface,
    then choosing between and predicting a singulation or a grasp,
    then the robot executing the predicted motion
    ...

    Attributes
    ----------
    robot : Robot_Interface
        low level interface that interacts with primitive HSR kinematics:
        http://hsr.io/
    ra : Robot_actions
        high level declutering task-specific interface that interacts
        with the HSR_CORE Robot_Interface:
        https://github.com/BerkeleyAutomation/HSR_CORE
    viz : boolean
        save images from perception pipeline

    """

    def __init__(self, viz=False, maskrcnn=False):
        """
        Class that runs decluttering task
        """
        self.robot = Robot_Interface()
        self.ra = Robot_Actions(self.robot)

        model_path = 'main/model/sim_then_real_on_real_hsr_7objects/output_inference_graph.pb'
        label_map_path = 'main/model/sim_then_real_on_real_hsr_7objects/object-detection.pbtxt'
        self.det = Detector(model_path, label_map_path)

        self.viz = viz

        # print("Finished Initialization")

    def run_grasp(self, bbox, c_img, col_img, workspace_img, d_img):
        '''
        Parameters
        ----------
        bbox : Bbox
            bounding box of object to
            be grasped
        c_img : cv2.img
            rgb image from robot
        col_img : ColorImage
            c_img wrapped in ColorImage
        workspace_img : BinaryImage
            c_img cropped to workspace
        d_img : cv2.img
            depth image from robot

        '''
        # print("Grasping a " + cfg.labels[bbox.label])
        try:
            group = bbox.to_group(c_img, col_img)
        except ValueError:
            return

        display_grasps(workspace_img, [group])

        self.ra.execute_grasp(group.cm, group.dir, d_img, class_num=bbox.label)

    def run_singulate(self, singulator, d_img):
        """
        execute the singulation strategy

        Parameters
        ----------
        singulator : Singulation
            class for handling singulation
            strategy
        d_img : cv2.img
            depth image from robot

        """
        # print("Singulating")
        waypoints, rot, free_pix = singulator.get_singulation()
        singulator.display_singulation()
        self.ra.execute_singulate(waypoints, rot, d_img)

    def get_bboxes_from_net(self, img, sess=None, save_img=False):
        """
        fetch bounding boxes from the object
        detection network

        '''
        Parameters
        ----------
        img : the image from the robot for prediction
        sess : tensorflow session for object recognition

        Returns
        -------
        bboxes : []
            list of Bbox objects labeled by hand
        vis_util_image : np.array()
            numpy array format of image

        """
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_dict, vis_util_image = self.det.predict(rgb_image, sess=sess)
        if save_img:
            plt.savefig('debug_imgs/predictions.png')
            plt.close()
            plt.clf()
            plt.cla()
        # img = cv2.imread(path)
        boxes = format_net_bboxes(output_dict, img.shape)
        return boxes, vis_util_image



    def find_grasps(self, groups, col_img):
        """
        find all lego blocks to grasp and singulate
        in groups

        '''
        Parameters
        ----------
        groups : []
            list of groups of clutter
        col_img : ColorImage
            rgb image wrapped in ColorImage class

        Returns
        -------
        to_grasp : []
            a list of groups of objects to grasp
        to_singulate : []
            list of groups of objects to
            singulate

        """
        to_grasp = []
        to_singulate = []
        for group in groups:
            inner_groups = grasps_within_pile(col_img.mask_binary(group.mask))

            if len(inner_groups) == 0:
                to_singulate.append(group)
            else:
                for in_group in inner_groups:
                    class_num = hsv_classify(col_img.mask_binary(in_group.mask))
                    color_name = class_num_to_name(class_num)
                    if color_name in cfg.HUES_TO_BINS:
                        lego_class_num = cfg.HUES_TO_BINS[color_name]
                        to_grasp.append((in_group, lego_class_num, color_name))
        return to_grasp, to_singulate


    def lego_demo(self):
        """
        demo that runs color based segmentation and declutters legos
        """

        # if true, use hard-coded deposit without AR markers
        hard_code = True

        # setup robot in front-facing start pose to take image of legos
        self.ra.go_to_start_pose()
        self.ra.set_start_position()
        self.ra.head_start_pose()
        c_img, d_img = self.robot.get_img_data()

        # self.ra.move_base(z=-2.7)



        with self.det.detection_graph.as_default():
            with tensorflow.Session() as sess:
                while not (c_img is None or d_img is None):
                    if self.viz:
                        path = 'debug_imgs/web.png'
                        cv2.imwrite(path, c_img)
                        time.sleep(2)

                    # crop image and generate binary mask
                    # main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz, task='lego_demo')

                    # bb; ipdb.set_trace()
                    main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz)
                    col_img = ColorImage(c_img)
                    workspace_img = col_img.mask_binary(main_mask)


                    bboxes, vis_util_image = self.get_bboxes_from_net(c_img, sess=sess)
                    if self.viz:
                        cv2.imwrite('debug_imgs/workspace_img.png', workspace_img.data)
                        cv2.imwrite('debug_imgs/object_detection.png', vis_util_image)
                    # import ipdb; ipdb.set_trace()

                    if len(bboxes) == 0:
                        print("Cleared the workspace")
                        print("Add more objects, then resume")
                        # import ipdb; ipdb.set_trace()
                    else:
                        if self.viz:
                            box_viz = draw_boxes(bboxes, c_img)
                            cv2.imwrite("debug_imgs/box.png", box_viz)
                        single_objs = find_isolated_objects_by_overlap(bboxes)
                        if len(single_objs) == 0:
                            single_objs = [bboxes[0]]
                            # import ipdb; ipdb.set_trace()
                            # single_objs = find_isolated_objects_by_distance(bboxes, col_img)

                        if len(single_objs) > 0:
                            to_grasp = select_first_obj(single_objs)

                            try:
                                # self.ra.execute_grasp(to_grasp.cm, to_grasp.dir, d_img, class_num=to_grasp.label)
                                self.run_grasp(to_grasp, c_img, col_img, workspace_img, d_img)
                            except Exception as ex:
                                print(ex)
                                self.ra.move_to_start()
                                self.ra.head_start_pose()
                                c_img, d_img = self.robot.get_img_data()
                                continue

                            # print(self.ra.get_start_position(), self.ra.get_position())
                            while True:
                                try:
                                    self.robot.body_neutral_pose()
                                    break
                                except Exception as ex:
                                    print(ex)

                            # import ipdb;ipdb.set_trace()
                            self.ra.move_base_abs()
                            # self.ra.move_to_start()
                            # self.ra.go_to_start_pose()

                            # deposit lego in it's corresponding colored bin
                            # offset_1 = 0.2
                            # offset_2 = 0.5
                            # offset_4 = 0.0
                            offset_0 = -1.0
                            offset_1 = -1.5
                            offset_2 = -2.1
                            offset_3 = -2.7

                            # offset_x = -0.3
                            # self.ra.move_base_abs(x=-0.0, y=0.0, z=-np.pi / 2.0)
                            print(to_grasp.label)

                            if hard_code:
                                if to_grasp.label == 2: #yellow
                                    # self.ra.move_base(z=-2.3)
                                    # self.ra.move_base(z=offset_2)
                                    # self.ra.move_base(z=-1.6)
                                    # self.ra.move_base(z=offset_1)
                                    # self.ra.move_base(x=0.4)
                                    self.ra.move_base_abs(x=-0.6, y=0.0, z=-np.pi/2.0)
                                    # import ipdb; ipdb.set_trace()
                                    # self.robot.move_in_cubby(x_pos=-0.65, z_pos=0.4)
                                    self.ra.deposit_in_cubby(x_pos=-0.13, z_pos=0.26, label=to_grasp.label)
                                    # self.ra.deposit_in_cubby(x_pos=0.01, z_pos=0.0, label=to_grasp.label)

                                elif to_grasp.label == 1: #blue
                                    # self.ra.move_base(z=-1.7)
                                    #  import ipdb; ipdb.set_trace()
                                    #  self.ra.move_base(z=1.5708)
                                    self.ra.move_base_abs(x=-0.3, y=0.0, z=-np.pi / 2.0)
                                    # self.ra.move_base(z=offset_1)
                                    # self.ra.move_base(x = 0.4)
                                    # self.ra.move_base_abs(x=-0.35, y=-0.4, z=offset_1)
                                    # import ipdb; ipdb.set_trace()

                                    self.ra.deposit_in_cubby(x_pos=-0.13, z_pos=0.26, label=to_grasp.label)
                                    # import ipdb; ipdb.set_trace()
                                elif to_grasp.label == 0: #red
                                    # self.ra.move_base(z=-1.1)
                                    # self.ra.move_base(z=offset_0)
                                    self.ra.move_base_abs(x=-0.0, y=0.0, z=-np.pi / 2.0)

                                    self.ra.deposit_in_cubby(x_pos=-0.13, z_pos=0.26, label=to_grasp.label)
                                    # import ipdb; ipdb.set_trace()
                                elif to_grasp.label == 3: #money
                                    # self.ra.move_base(z=offset_1)
                                    # self.ra.move_base(x = 0.4)
                                    # self.ra.move_base_abs(x=-0.35, y=-0.4, z=offset_1)

                                    self.robot.move_in_cubby(x_pos=-0.34, z_pos=0.0)
                                    self.ra.move_base_abs(x=-0.3, y=0.0, z=-np.pi / 2.0)
                                    self.ra.move_base(x=0.1)
                                    self.robot.open_gripper()
                                    self.robot.close_gripper()
                                    # if label == 3 or label == 4 or label == 5:
                                    #     self.robot.move_base(x=-0.30)
                                    self.ra.move_base(x=-0.2)
                                    self.ra.move_base_abs()
                                    self.robot.body_neutral_pose()
                                    self.robot.body_start_pose()
                                    time.sleep(2)


                                    # self.robot.move_in_cubby(y_pos=-0.1, z_pos=0.0)

                                    # self.ra.deposit_in_cubby(x_pos=-0.1, z_pos=0.0, label=to_grasp.label)
                                elif to_grasp.label == 4: #towel

                                    # self.ra.move_base_abs(x=-0.6, y=0.0, z=-np.pi / 2.0)

                                    self.robot.move_in_cubby(x_pos=-0.34, z_pos=0.0)
                                    self.ra.move_base_abs(x=-0.6, y=0.0, z=-np.pi / 2.0)
                                    self.ra.move_base(x=0.1)
                                    self.robot.open_gripper()
                                    self.robot.close_gripper()
                                    # if label == 3 or label == 4 or label == 5:
                                    #     self.robot.move_base(x=-0.30)
                                    self.ra.move_base(x=-0.2)
                                    self.ra.move_base_abs()
                                    self.robot.body_neutral_pose()
                                    self.robot.body_start_pose()
                                    time.sleep(2)

                                    # self.ra.move_base(z=offset_1)
                                    # self.ra.move_base(x = 0.4)
                                    # self.ra.move_base_abs(x=-0.65, y=-0.4, z=offset_1)
                                    # self.ra.deposit_in_cubby(x_pos=-0.34, z_pos=0.0, label=to_grasp.label)
                                elif to_grasp.label == 5: #tongs
                                    # self.ra.move_base_abs(x=-0.0, y=0.0, z=-np.pi / 2.0)


                                    self.robot.move_in_cubby(x_pos=-0.34, z_pos=0.0)
                                    self.ra.move_base_abs(x=-0.0, y=0.0, z=-np.pi / 2.0)
                                    self.ra.move_base(x=0.1)
                                    self.robot.open_gripper()
                                    self.robot.close_gripper()
                                    # if label == 3 or label == 4 or label == 5:
                                    #     self.robot.move_base(x=-0.30)
                                    self.ra.move_base(x=-0.2)
                                    self.ra.move_base_abs()
                                    self.robot.body_neutral_pose()
                                    self.robot.body_start_pose()
                                    time.sleep(2)

                                    # self.ra.move_base(z=offset_1)
                                    # self.ra.move_base(x = 0.45)
                                    # self.ra.deposit_in_cubby(x_pos=-0.34, z_pos=0.0, label=to_grasp.label)
                                    # import ipdb; ipdb.set_trace()
                                # self.ra.move_base(z=-1.6)
                                # self.robot.body_neutral_pose()
                                # self.robot.body_neutral_pose() #probably not required.
                                # self.ra.move_base(z=1.6)
                                # if to_grasp.label == 2:
                                #     # self.ra.move_base(x=0.6)
                                #     # self.ra.move_base(z=1.6)
                                #     # self.ra.move_base(x=0.4)
                                #     self.ra.move_base(z=-offset_1)
                                # elif to_grasp.label == 1:
                                #     # self.ra.move_base(x=0.3)
                                #     self.ra.move_base(z=-offset_1)
                                # elif to_grasp.label == 0:
                                #     self.ra.move_base(z=-offset_1)
                                # elif to_grasp.label == 3:
                                #     self.ra.move_base(z=-offset_1)
                                # elif to_grasp.label == 4:
                                #     self.ra.move_base(z=-offset_1)
                                # elif to_grasp.label == 5:
                                #     self.ra.move_base(z=-offset_1)
                            else:
                                try:
                                    self.ra.deposit_obj(to_grasp.label)
                                except Exception as ex:
                                    print(ex)
                            # self.run_grasp(to_grasp, c_img, col_img, col_img, d_img)
                        else:


                            groups = [box.to_group(c_img, col_img) for box in bboxes]
                            groups = merge_groups(groups, cfg.DIST_TOL)
                            singulator = Singulation(col_img, main_mask, [g.mask for g in groups])
                            self.run_singulate(singulator, d_img)

                        # return to the position it was in before grasping
                        # self.ra.go_to_start_pose()
                        self.ra.move_base_abs()
                        self.ra.move_to_start() #probably not required.
                        self.ra.head_start_pose()

                    c_img, d_img = self.robot.get_img_data()


    def test(self):
        c_img = cv2.imread('debug_imgs/web.png')
        main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz)
        col_img = ColorImage(c_img)
        workspace_img = col_img.mask_binary(main_mask)
        groups = run_connected_components(workspace_img, viz=self.viz)
        to_grasp, to_singulate = self.find_grasps(groups, col_img)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = DeclutterDemo(viz=True)
    # rospy.spin()
    task.lego_demo()
