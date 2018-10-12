import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

# import tpc package modules
from tpc.perception.cluster_registration import run_connected_components, display_grasps, class_num_to_name, grasps_within_pile, hsv_classify
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance
from tpc.manipulation.robot_actions import Robot_Actions
from tpc.data.helper import Helper
from tpc.data.data_logger import DataLogger
import tpc.config.config_tpc as cfg
# from tpc.detection.detector import Detector
# from tpc.detection.maskrcnn_detection import detect

# import high-level robot interface 
if cfg.robot_name == "hsr":
    from core.hsr_robot_interface import Robot_Interface
elif cfg.robot_name == "fetch":
    from core.fetch_robot_interface import Robot_Interface
elif cfg.robot_name is None:
    from tpc.offline.robot_interface import Robot_Interface

# import web labeler
sys.path.append("hsr_web/")
from web_labeler import Web_Labeler

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
        # self.helper = Helper(cfg)
        self.ra = Robot_Actions(self.robot)
        # # self.dl = DataLogger("stats_data/model_base", cfg.EVALUATE)
        # self.web = Web_Labeler(cfg.NUM_ROBOTS_ON_NETWORK)

        # self.maskrcnn = maskrcnn
        # if not self.maskrcnn:
        #     model_path = 'main/model/output_inference_graph.pb'
        #     label_map_path = 'main/model/object-detection.pbtxt'
        #     self.det = Detector(model_path, label_map_path)

        self.viz = viz

        print("Finished Initialization")

    def run_grasp(self, bbox, c_img, col_img, workspace_img, d_img):
        """
        execute the grasp strategy

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

        """
        print("Grasping a " + cfg.labels[bbox.label])
        try:
            group = bbox.to_group(c_img, col_img)
        except ValueError:
            return

        display_grasps(workspace_img, [group])

        self.ra.execute_grasp(group.cm, group.dir, d_img, class_num=bbox.label)

    def run_singulate(self, singulator, d_img):
        """
        execute the singulation strategy
        
        '''
        Parameters
        ----------
        singulator : Singulation
            class for handling singulation
            strategy
        d_img : cv2.img
            depth image from robot

        """
        print("Singulating")
        waypoints, rot, free_pix = singulator.get_singulation()
        singulator.display_singulation()
        self.ra.execute_singulate(waypoints, rot, d_img)

    def get_bboxes_from_net(self, path):
        """
        fetch bounding boxes from the object 
        detection network

        '''
        Parameters
        ----------
        path : String
            the filepath of the image from the robot
            to run through the network

        Returns
        -------
        bboxes : []
            list of Bbox objects labeled by hand
        vis_util_image : np.array()
            numpy array format of image

        """
        if not self.maskrcnn:
            output_dict, vis_util_image = self.det.predict(path)
            plt.savefig('debug_imgs/predictions.png')
            plt.close()
            plt.clf()
            plt.cla()
        else:
            output_dict = detect(path)
            vis_util_image = None
        img = cv2.imread(path)
        boxes = format_net_bboxes(output_dict, img.shape)
        return boxes, vis_util_image

    def get_bboxes_from_web(self, path):
        """
        fetch bounding boxes from the web labeler
        
        '''
        Parameters
        ----------
        path : String
            the filepath of the image from the robot
            to label

        Returns
        -------
        bboxes : []
            list of Bbox objects labeled by hand
        vectors : []
            list of vectors to use for singulation
        
        """
        labels = self.web.label_image(path)

        objs = labels['objects']
        bboxes = []
        vectors = []
        for obj in objs:
            if obj['motion'] == 1:
                coords = obj['coords']
                p0 = [coords[1], coords[0]]
                p1 = [coords[3], coords[2]]
                vectors.append(([p0, p1], obj['class']))
            else:
                bboxes.append(Bbox(obj['coords'], obj['class']))
        return bboxes, vectors

    def determine_to_ask_for_help(self,bboxes,col_img):
        """
        Determine whether to ask for help from the
        web labeler by looking at the overlap and
        separation between objects
        
        '''
        Parameters
        ----------
        bboxes : []
            list of Bbox objects
        col_img : ColorImage
            rgb image wrapped in ColorImage class

        Returns
        -------
        boolean
            True if ask for help, false otherwise

        """
        bboxes = deepcopy(bboxes)
        col_img = deepcopy(col_img)

        single_objs = find_isolated_objects_by_overlap(bboxes)

        if len(single_objs) > 0:
            return False
        else:
            single_objs = find_isolated_objects_by_distance(bboxes, col_img)
            return len(single_objs) == 0

    def get_bboxes(self, path,col_img):
        """
        get all the bounding boxes around objects
        in the image, also finds push strategies

        '''
        Parameters
        ----------
        path : String
            file path to save debug image
        col_img : ColorImage
            rgb image wrapped in ColorImage class

        Returns
        -------
        boxes : []
            a list of Bbox objects
        vectors : []
            a list of vectors to use for
            singulation

        """
        boxes, vis_util_image = self.get_bboxes_from_net(path)
        vectors = []

        #low confidence or no objects
        if len(boxes) == 0 or self.determine_to_ask_for_help(boxes,col_img):
            self.helper.asked = True
            self.helper.start_timer()
            boxes, vectors = self.get_bboxes_from_web(path)
            self.helper.stop_timer()
            # self.dl.save_stat("duration", self.helper.duration)
            # self.dl.save_stat("num_online", cfg.NUM_ROBOTS_ON_NETWORK)

        return boxes, vectors, vis_util_image

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

    def tools_demo(self):
        """
        demo that runs objects detection and declutters tools,
        runs until workspace is clear

        """
        self.ra.go_to_start_pose()
        # self.ra.set_start_position()
        # self.ra.head_start_pose()
        c_img, d_img = self.robot.get_img_data()

        while not (c_img is None or d_img is None):
            path = 'debug_imgs/web.png'
            cv2.imwrite(path, c_img)
            time.sleep(2) #make sure new image is written before being read

            # crop image and generate binary mask
            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            # get the bounding boxes of all the tools
            bboxes, vectors, vis_util_image = self.get_bboxes(path,col_img)

            if len(bboxes) > 0:

                # visualize bounding boxes
                box_viz = draw_boxes(bboxes, c_img)
                cv2.imwrite("debug_imgs/box.png", box_viz)

                # find groups of cluttered tools
                single_objs = find_isolated_objects_by_overlap(bboxes)
                grasp_success = 1.0

                # find invdividual tools in clutter
                if len(single_objs) == 0:
                    single_objs = find_isolated_objects_by_distance(bboxes, col_img)

                # choose grasping strategy
                if len(single_objs) > 0:
                    to_grasp = select_first_obj(single_objs)
                    singulation_time = 0.0
                    self.run_grasp(to_grasp, c_img, col_img, workspace_img, d_img)
                    # grasp_success = self.dl.record_success("grasp", other_data=[c_img, vis_util_image, d_img])
                    self.ra.move_base(z=-1.55)
                    self.ra.deposit_obj(to_grasp.label)
                    self.ra.move_base(z=1.55)

                # choose singulation strategy
                else:
                    #for accurate singulation should have bboxes for all
                    groups = [box.to_group(c_img, col_img) for box in bboxes]
                    groups = merge_groups(groups, cfg.DIST_TOL)
                    singulator = Singulation(col_img, main_mask, [g.mask for g in groups])
                    self.run_singulate(singulator, d_img)
                    sing_start = time.time()
                    # singulation_success = self.dl.record_success("singulation", other_data=[c_img, vis_util_image, d_img])
                    singulation_time = time.time()-sing_start

                # evaluate reward function for grasp/singulation
                if cfg.EVALUATE:
                    reward = self.helper.get_reward(grasp_success,singulation_time)
                    # self.dl.record_reward(reward)

            # singulate if pushing policies exist
            elif len(vectors) > 0:
                waypoints, class_labels = vectors[0]
                rot = 0
                singulator = Singulation(col_img, main_mask, [], goal_p=waypoints[-1], waypoints=waypoints, gripper_angle=rot)
                self.run_singulate(singulator, d_img)

            # all tools cleared in workspace
            else:
                print("Cleared the workspace")
                print("Add more objects, then resume")
                IPython.embed()

            # return to the starting state
            # self.ra.move_to_start()
            # self.ra.head_start_pose()

            self.ra.go_to_start_position()

            self.ra.go_to_start_pose()
            c_img, d_img = self.robot.get_img_data()

    def lego_demo(self):
        """
        demo that runs color based segmentation and declutters legos,
        runs until workspace is clear

        """

        # if true, use hard-coded deposit without AR markers
        hard_code = True

        # setup robot in front-facing start pose to take image of legos
        self.ra.go_to_start_pose()
        self.ra.set_start_position()
        self.ra.head_start_pose()
        c_img, d_img = self.robot.get_img_data()

        while not (c_img is None or d_img is None):
            if self.viz:
                path = 'debug_imgs/web.png'
                cv2.imwrite(path, c_img)
                time.sleep(2)

            # crop image and generate binary mask
            main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            # find groups of legos as individual seg masks
            groups = run_connected_components(workspace_img, viz=self.viz)
            valid_groups = []

            # extract only legos in valid color set
            for group in groups:
                class_num = hsv_classify(col_img.mask_binary(group.mask))
                color_name = class_num_to_name(class_num)
                if color_name in cfg.HUES_TO_BINS:
                    valid_groups.append(group)

            # workspace has legos
            if len(valid_groups) > 0:

                # find legos to be grasped and singulated within cluterred groups
                to_grasp, to_singulate = self.find_grasps(valid_groups, col_img)

                # choose grasping strategy
                if len(to_grasp) > 0:

                    # find closest lego to grasp
                    to_grasp.sort(key=lambda g:-1 * g[0].cm[0])
                    if not cfg.CHAIN_GRASPS:
                        to_grasp = to_grasp[0:1]
                    if self.viz:
                        display_grasps(workspace_img, [g[0] for g in to_grasp], name="debug_imgs/grasp")

                    # save the color and label of lego for depositing later
                    group = to_grasp[0][0]
                    label = to_grasp[0][1]
                    color = to_grasp[0][2]
                    print("Grasping a " + color + " lego", label)

                    # attempt to execute grasp of lego
                    try:
                        self.ra.execute_grasp(group.cm, group.dir, d_img, class_num=label)
                    except Exception as ex:
                        print(ex)
                        self.ra.move_to_start()
                        self.ra.head_start_pose()
                        c_img, d_img = self.robot.get_img_data()
                        continue
                        
                    print(self.ra.get_start_position(), self.ra.get_position())
                    while True:
                        try:
                            self.robot.body_neutral_pose()
                            break
                        except Exception as ex:
                            print(ex)

                    # return to the position it was in before grasping
                    # self.ra.go_to_start_pose()
                    self.ra.move_to_start()

                    # deposit lego in it's corresponding colored bin
                    if hard_code:
                        if label == 2:
                            self.ra.move_base(z=-2.3)
                            # self.ra.move_base(x=-0.4)
                            # self.ra.move_base(z=-1.6)
                        if label == 1:
                            self.ra.move_base(z=-1.7)
                        if label == 0:
                            self.ra.move_base(z=-1.1)
                        # self.ra.move_base(z=-1.6)
                        # self.robot.body_neutral_pose()
                        self.robot.body_neutral_pose()
                        self.ra.deposit_in_cubby()
                        # self.ra.move_base(z=1.6)
                        if label == 2:
                            # self.ra.move_base(x=0.6)
                            # self.ra.move_base(z=1.6)
                            # self.ra.move_base(x=0.4)
                            self.ra.move_base(z=2.3)
                        if label == 1:
                            # self.ra.move_base(x=0.3)
                            self.ra.move_base(z=1.7)
                        if label == 0:
                            self.ra.move_base(z=1.1)
                    else:
                        try:
                            self.ra.deposit_obj(label)
                        except Exception as ex:
                            print(ex)

                # choose singulation strategy
                else:
                    # singulator = Singulation(col_img, main_mask, [g.mask for g in to_singulate])
                    # self.run_singulate(singulator, d_img)
                    # self.ra.go_to_start_pose()
                    print('Singulating')
                    group = valid_groups[0]
                    try:
                        self.ra.spread_singulate(group.cm, group.dir, d_img)
                        # self.ra.l_singulate(group.cm, group.dir, d_img)
                    except Exception as ex:
                        print(ex)
                    self.ra.go_to_start_pose()

            # workspace is clear
            else:
                print("Cleared the workspace")
                print("Add more objects, then resume")
                IPython.embed()

            # return to starting position and begin new iteration
            self.ra.move_to_start()
            # self.ra.go_to_start_pose()
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
    simple = True
    if simple:
        task.lego_demo()
        # task.test()
    else:
        task.tools_demo()
