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
import math
import os
import time

import importlib
import matplotlib.pyplot as plt


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

from hsr_core.sensors import RGBD

import tf2_ros

import geometry_msgs



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

        #model_path = 'main/model/sim_then_real_on_real_4objects/output_inference_graph.pb'
        #label_map_path = 'main/model/sim_then_real_on_real_4objects/object-detection.pbtxt'
        #self.det = Detector(model_path, label_map_path)

        self.viz = viz

        self.cam = RGBD()

        self.tfBuffer = tf2_ros.Buffer()

    def run_grasp_gqcnn(self, c_img, d_img, number_failed):
        import json
        import os


        from PIL import Image

        from autolab_core import RigidTransform, YamlConfig, Logger
        from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
        from visualization import Visualizer2D as vis

        from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction
        from gqcnn.utils import GripperMode, NoValidGraspsException

        segmask_filename = None
        camera_intr_filename = None
        model_dir = None
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                '..',
                                                'gqcnn/cfg/examples/dex-net_4.0_hsr.yaml')
        fully_conv = None

        assert not (fully_conv and depth_im_filename is not None and segmask_filename is None), 'Fully-Convolutional policy expects a segmask.'

        if fully_conv and segmask_filename is None:
            segmask_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                '..',
                                                'gqcnn/data/examples/clutter/primesense/segmask_0.png')
        if camera_intr_filename is None:
            camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                '..',
                                                'gqcnn/data/calib/primesense/primesense.intr')    
        if config_filename is None:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                '..',
                                                'gqcnn/cfg/examples/fc_policy.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                '..',
                                                'gqcnn/cfg/examples/policy.yaml')
       
        # read config
        config = YamlConfig(config_filename)
        inpaint_rescale_factor = config['inpaint_rescale_factor']
        policy_config = config['policy']

        # set model if provided and make relative paths absolute
        if model_dir is not None:
            policy_config['metric']['gqcnn_model'] = model_dir
        if 'gqcnn_model' in policy_config['metric'].keys() and not os.path.isabs(policy_config['metric']['gqcnn_model']):
            policy_config['metric']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                  '..',
                                                                  '../gqcnn',
                                                                  policy_config['metric']['gqcnn_model'])

        # setup sensor
        camera_intr = CameraIntrinsics.load(camera_intr_filename)
            
        # read images
        depth_data = self.focus_on_target_zone(d_img)


        depth_im = DepthImage(depth_data, frame=camera_intr.frame)
        rgb_img = c_img
        #rgb_img_np = np.array(rgb_img).astype(np.uint8)
        #rgb_img_cropped = rgb_img.crop((245, 150, 420, 310))
        #plt.imshow(depth_data)
        #plt.show()
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8), frame=camera_intr.frame)

        
        # optionally read a segmask
        segmask = None
        if segmask_filename is not None:
            segmask = BinaryImage.open(segmask_filename)
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        if segmask is None:
            segmask = valid_px_mask
        else:
            segmask = segmask.mask_binary(valid_px_mask)
        
        # inpaint
        depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
            
        if 'input_images' in policy_config['vis'].keys() and policy_config['vis']['input_images']:
            vis.figure(size=(10,10))
            num_plot = 1
            if segmask is not None:
                num_plot = 2
            vis.subplot(1,num_plot,1)
            vis.imshow(depth_im)
            if segmask is not None:
                vis.subplot(1,num_plot,2)
                vis.imshow(segmask)
            vis.show()
            
        # create state
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

        # set input sizes for fully-convolutional policy
        if fully_conv:
            policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
            policy_config['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]

        # init policy
        if fully_conv:
            #TODO: @Vishal we should really be doing this in some factory policy
            if policy_config['type'] == 'fully_conv_suction':
                policy = FullyConvolutionalGraspingPolicySuction(policy_config)
            elif policy_config['type'] == 'fully_conv_pj':
                policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
            else:
                raise ValueError('Invalid fully-convolutional policy type: {}'.format(policy_config['type']))
        else:
            policy_type = 'cem'
            if 'type' in policy_config.keys():
                policy_type = policy_config['type']
            if policy_type == 'ranking':
                policy = RobustGraspingPolicy(policy_config)
            elif policy_type == 'cem':
                policy = CrossEntropyRobustGraspingPolicy(policy_config)
            else:
                raise ValueError('Invalid policy type: {}'.format(policy_type))

        
        # query policy
        policy_start = time.time()
        action = policy(state)
        # grasp_center[x,y] in image frame
        grasp_center = [action.grasp.center[0], action.grasp.center[1]]
        grasp_angle = action.grasp.angle
        grasp_depth_m = action.grasp.depth
        grasp_height_offset = action.grasp.height_offset
        grasp_width = action.grasp.width
        print('grasp width %f' %(grasp_width))

        # ignore corrupted depth images
        if 0.7 < grasp_depth_m < 1.05:
            pass
        else:
            print('invalid depth image')
            return number_failed + 1

        rgb_img = np.asarray(c_img)
        color_im = ColorImage(rgb_img,
                                frame=camera_intr.frame, encoding='rgb8')
        color_im = color_im.rgb2bgr()

        TARGET_DIR = '/home/benno/experiments/hsr/gqcnn/10_15_elevation_v2/'
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)

        #files_in_TARGET_DIR = os.listdir(TARGET_DIR)
        files_in_TARGET_DIR = [os.path.join(TARGET_DIR, filename) for filename in os.listdir(TARGET_DIR)]
        if not files_in_TARGET_DIR:
            new_file_number = 0
        else:
            last_added_file = max(files_in_TARGET_DIR, key=os.path.getmtime)
            last_file_number = last_added_file[-7:-4]
            new_file_number = int(last_file_number) + 1

        # vis final grasp
        policy_config['vis']['final_grasp'] = False
        if policy_config['vis']['final_grasp']:
            vis.figure(size=(40,40))
            vis.subplot(1,2,1)
            vis.imshow(rgbd_im.depth,
                       vmin=policy_config['vis']['vmin'],
                       vmax=policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.subplot(1,2,2)
            vis.imshow(color_im)
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            #vis.savefig(TARGET_DIR + 'final_grasp_' + str(new_file_number).zfill(3))
            vis.show()


        #self.get_height_line(grasp_center, grasp_angle, d_img)

        # execute planned grasp with hsr interface
        #self.execute_gqcnn(grasp_center, grasp_angle, d_img*1000)

        # execute 2DOF grasp
        self.execute_gqcnn_2DOF(grasp_center, grasp_depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img*1000)
        return 0

    def focus_on_target_zone(self, d_img):
        d_img[:, :245] = 0
        d_img[:, 420:] = 0
        d_img[:150, :] = 0
        d_img[310:, :] = 0
        return d_img

    def execute_gqcnn(self, grasp_center, grasp_angle, depth_image_mm):
        grasp_direction = np.array([math.sin(grasp_angle), math.cos(grasp_angle)])
        grasp_direction_normalized = grasp_direction / np.linalg.norm(grasp_direction)
        self.ra.execute_grasp(grasp_center, grasp_direction_normalized, depth_image_mm, 0, 500.0)

    def execute_gqcnn_2DOF(self, grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img):
        self.ra.execute_2DOF_grasp(grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img)


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
        # display_grasps(workspace_img, [group])
        self.ra.execute_grasp(group.cm, group.dir, d_img, bbox.label, 500.0)

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

    def lego_demo(self, number_failed):
        self.ra.go_to_start_pose()
        time.sleep(1)
        c_img, d_img = self.robot.get_img_data()
        while (c_img is None or d_img is None):
            c_img, d_img = self.robot.get_img_data()

        # convert depth image from mm to m because dexnet uses m
        depth_image_mm = np.asarray(d_img[:,:])
        depth_image_m = depth_image_mm/1000

        number_failed = self.run_grasp_gqcnn(c_img, depth_image_m, number_failed)
        return number_failed
        
    
    def lego_demo_old(self):
        """
        demo that runs color based segmentation and declutters legos
        """

        # if true, use hard-coded deposit without AR markers
        hard_code = True

        # setup robot in front-facing start pose to take image of legos
        #self.ra.go_to_start_pose()
        #self.ra.set_start_position()
        self.ra.head_start_pose()
        c_img, d_img = self.robot.get_img_data()


        # self.ra.move_base(z=-2.7)

        # import ipdb; ipdb.set_trace()
        while c_img is None or d_img is None:
            c_img, d_img = self.robot.get_img_data()
        with self.det.detection_graph.as_default():
            with tensorflow.Session() as sess:
                while not (c_img is None or d_img is None):
                    # if self.viz:
                    #     path = 'debug_imgs/web.png'
                    #     cv2.imwrite(path, c_img)
                    #     time.sleep(2)

                    # crop image and generate binary mask
                    # main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz, task='lego_demo')
                    # import ipdb; ipdb.set_trace()
                    main_mask = crop_img(c_img, simple=True, arc=False, viz=self.viz)
                    col_img = ColorImage(c_img)
                    workspace_img = col_img.mask_binary(main_mask)

                    # cv2.imwrite('debug_imgs/workspace_img.png', workspace_img.data)
                    bboxes, vis_util_image = self.get_bboxes_from_net(c_img, sess=sess)
                    if len(bboxes) == 0:
                        print("Cleared the workspace")
                        print("Add more objects, then resume")
                        return
                        #import ipdb; ipdb.set_trace()
                    else:
                        # box_viz = draw_boxes(bboxes, c_img)
                        # cv2.imwrite("debug_imgs/box.png", box_viz)
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
                                #self.ra.move_to_start()
                                #self.ra.head_start_pose()
                                c_img, d_img = self.robot.get_img_data()
                                continue
                        else:
                            groups = [box.to_group(c_img, col_img) for box in bboxes]
                            groups = merge_groups(groups, cfg.DIST_TOL)
                            singulator = Singulation(col_img, main_mask, [g.mask for g in groups])
                            self.run_singulate(singulator, d_img)

                        # return to the position it was in before grasping
                        # self.ra.go_to_start_pose()
                        varr = raw_input()
                        self.ra.robot.open_gripper()
                        self.ra.robot.omni_base.go_abs(0,0,0,0)
                        self.ra.robot.whole_body.move_to_joint_positions({'arm_flex_joint': -0.005953039901891888,
                                                            'arm_lift_joint': 3.5673664703075522e-06,
                                                            'arm_roll_joint': -1.6400026753088877,
                                                            'head_pan_joint': 1.59,
                                                            'head_tilt_joint': -1.15,
                                                            'wrist_flex_joint': -1.570003402348724,
                                                            'wrist_roll_joint': 0})
                        return
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

    number_failed = 0
    while number_failed <= 3:
        print('Starting new run with %f fails in a row now' %(number_failed))
        task = DeclutterDemo(viz=True)
        # rospy.spin()
        number_failed = task.lego_demo(number_failed)
    print('Nothing to grasp, demo done.')
