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
import json

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


from object_detection.utils import visualization_utils as vis_util



img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')


class SurfaceDeclutter():

    def __init__(self):
        init_start = time.time()
        """
        Class that runs surface decluttering task
        """
        self.robot = Robot_Interface()
        self.ra = Robot_Actions(self.robot)

        #model_path = '/nfs/diskstation/ajaytanwani/trained_models/real_hsr_5_objects_v2'
        model_path = '/nfs/diskstation/ajaytanwani/trained_models/real_on_real_fpn/output_inference_graph.pb'
        label_map_path = 'main/model/sim_then_labeled_dann_on_real/object-detection.pbtxt'
        self.labels = { 0: 'Tool', 1: 'Scrap', 2: 'Tube', 3: 'Box', 4: 'Plastic'}
        self.det = Detector(model_path, label_map_path)
        self.cam = RGBD()
        self.tfBuffer = tf2_ros.Buffer()
        init_end = time.time()
        self.TARGET_DIR = '/home/benno/experiments/hsr/surface_decluttering/isolated_objects_v2/'
        if not os.path.exists(self.TARGET_DIR):
           os.makedirs(self.TARGET_DIR)

        files_in_TARGET_DIR = [os.path.join(self.TARGET_DIR, filename) for filename in os.listdir(self.TARGET_DIR)]
        if not files_in_TARGET_DIR:
            self.new_file_number = 0
        else:
            last_added_file = max(files_in_TARGET_DIR, key=os.path.getmtime)
            last_file_number = last_added_file[-7:-4]
            self.new_file_number = int(last_file_number) + 1
        self.init_time = init_end-init_start

    def run_grasp_gqcnn(self, c_img, d_img, object_label):
        plan_start = time.time()
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
        depth_im = DepthImage(d_img, frame=camera_intr.frame)
        rgb_img = np.asarray(c_img)
        color_im = ColorImage(rgb_img,
                                frame=camera_intr.frame, encoding='rgb8')
        color_im = color_im.rgb2bgr()

        
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
            
        #vis.figure(size=(10,10))
        #num_plot = 1
        #if segmask is not None:
        #    num_plot = 2
        #vis.subplot(1,num_plot,1)
        #vis.imshow(depth_im)
        #vis.subplot(1,num_plot,2)
        #vis.imshow(color_im)
        #vis.savefig(self.TARGET_DIR + 'input_images_' + str(self.new_file_number).zfill(3))
        #vis.show()
            
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
        if action is None:
            plan_end = time.time()
            self.plan_grasp_time = plan_end - plan_start
            return 0
        # grasp_center[x,y] in image frame
        grasp_center = [action.grasp.center[0], action.grasp.center[1]]
        grasp_angle = action.grasp.angle
        grasp_depth_m = action.grasp.depth
        grasp_height_offset = action.grasp.height_offset
        grasp_width = action.grasp.width

        # ignore corrupted depth images
        if 0.7 < grasp_depth_m < 1.05:
            pass
        else:
            print('invalid depth image')
            return 0

        plan_end = time.time()
        self.plan_grasp_time = plan_end - plan_start
        print('Planning took %.2f seconds' %(plan_end - plan_start))
        # vis final grasp
        #policy_config['vis']['final_grasp'] = False
        if policy_config['vis']['final_grasp']:
            vis.figure(size=(40,40))
            vis.subplot(1,2,1)
            vis.imshow(rgbd_im.depth,
                       vmin=policy_config['vis']['vmin'],
                       vmax=policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.subplot(1,2,2)
            vis.imshow(rgbd_im.color)
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.savefig(self.TARGET_DIR + 'final_grasp_' + str(self.new_file_number).zfill(3))
            vis.show()
        #self.new_file_number += 1

        # execute planned grasp with hsr interface
        #self.execute_gqcnn(grasp_center, grasp_angle, d_img*1000)
        # execute 2DOF grasp
        self.execute_gqcnn_2DOF(grasp_center, grasp_depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img*1000, object_label)
        return 1



    def focus_on_target_zone(self, d_img, bbox):
        d_img_copy = d_img.copy()
        d_img_copy[:, :bbox.xmin] = 0
        d_img_copy[:, bbox.xmax:] = 0
        d_img_copy[:bbox.ymin, :] = 0
        d_img_copy[bbox.ymax:, :] = 0
        return d_img_copy

    def execute_gqcnn(self, grasp_center, grasp_angle, depth_image_mm):
        grasp_direction = np.array([math.sin(grasp_angle), math.cos(grasp_angle)])
        grasp_direction_normalized = grasp_direction / np.linalg.norm(grasp_direction)
        self.ra.execute_grasp(grasp_center, grasp_direction_normalized, depth_image_mm, 0, 500.0)

    def execute_gqcnn_2DOF(self, grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img, object_label):
        self.grasp_time, self.drop_time, self.go_to_start_time = self.ra.execute_2DOF_grasp(grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img, object_label)

    def get_bboxes_from_net(self, img, sess=None, file_number=None):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_dict, vis_util_image = self.det.predict(rgb_image, sess=sess, file_number=file_number)
        boxes = format_net_bboxes(output_dict, img.shape)
        return boxes, vis_util_image

    def read_RGBD_image(self):
        c_img, d_img = self.robot.get_img_data()
        while (c_img is None or d_img is None):
            c_img, d_img = self.robot.get_img_data()
        # Crop images to neglect part with robot arm blocking the view
        d_img[320:,:] = 0
        c_img[320:,:,:] = 0
        return c_img, d_img

    def declutter(self):
        self.ra.go_to_start_pose()
        time.sleep(1)
        c_img, d_img = self.read_RGBD_image()
        # convert depth image from mm to m because dexnet uses m
        depth_image_mm = np.asarray(d_img[:,:])
        depth_image_m = depth_image_mm/1000
        number_failed = self.run_grasp_gqcnn(c_img, depth_image_m)
        return

    def grasp_nearest_bbox_with_highest_confidence(self, c_img, d_img, bboxes):
        y_max = 0
        for bbox in bboxes:
            if bbox.ymax >= y_max:
                y_max = bbox.ymax
                target_bbox = bbox
        max_prob = target_bbox.prob
        for bbox in bboxes:
            if bbox.ymax > y_max - 50 and bbox.prob > max_prob:
                max_prob = bbox.prob
                target_bbox = bbox
        rgb_image = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
        d_img_focused = self.focus_on_target_zone(d_img, target_bbox)
        depth_image_mm = np.asarray(d_img_focused[:,:])
        depth_image_m = depth_image_mm/1000
        immg = target_bbox.draw(c_img)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(immg)
        plt.subplot(1,2,2)
        plt.imshow(d_img_focused)
        plt.savefig(self.TARGET_DIR + 'target_bbox_' + str(self.new_file_number).zfill(3))
        plt.close()
        grasp_found = self.run_grasp_gqcnn(c_img, depth_image_m, target_bbox.label)
        return grasp_found, target_bbox
        
    
    def segment(self, number_failed):
        self.ra.go_to_start_pose()
        # This time delay is needed to avoid the robot taking the RGB image too early
        time.sleep(2)
        read_img_start = time.time()
        c_img, d_img = self.read_RGBD_image()
        while c_img is None or d_img is None:
            c_img, d_img = self.read_RGBD_image()
        read_img_end = time.time()
        self.read_images_time = read_img_end - read_img_start
        rgb_image = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(rgb_image)
        plt.subplot(1,2,2)
        plt.imshow(d_img, cmap=plt.cm.gray_r)
        plt.savefig(self.TARGET_DIR + 'input_images_' + str(self.new_file_number).zfill(3))
        plt.close()
        segment_start = time.time()
        with self.det.detection_graph.as_default():
            with tensorflow.Session() as sess:
                main_mask = crop_img(c_img, simple=True, arc=False)
                col_img = ColorImage(c_img)
                workspace_img = col_img.mask_binary(main_mask)

                bboxes, vis_util_image = self.get_bboxes_from_net(c_img, sess=sess, file_number=self.new_file_number)
                bboxes = [bbox for bbox in bboxes if bbox.prob > 0.6]
                segment_end = time.time()
                self.segment_time = segment_end - segment_start
                if len(bboxes) == 0:
                    print("Cleared the workspace")
                    print("Add more objects, then resume")
                    self.new_file_number += 1
                    return 3
                else:
                    grasp_found = False
                    while not grasp_found:
                        grasp_found, target_bbox = self.grasp_nearest_bbox_with_highest_confidence(c_img, d_img, bboxes)
                        bboxes = [bbox for bbox in bboxes if bbox != target_bbox]
                        if not grasp_found:
                            print('Removing ghost bbox')
                    self.new_file_number += 1
                    return number_failed




if __name__ == "__main__":
    number_failed = 0
    task = SurfaceDeclutter()
    while number_failed <= 2:
    #    print('Starting new run with %d fails in a row now' %(number_failed))
        # rospy.spin()
        number_failed = task.segment(number_failed)
        file = open('/home/benno/experiments/hsr/surface_decluttering/isolated_objects_v1.txt', 'a')
        file.write('\r\nTrial No. ' + str(task.new_file_number - 1) + ' ' +
                    'Initialization: ' + str(round(task.init_time,2)) + 's: ' +
                    'Read images:' + str(round(task.read_images_time,2)) + 's ' +
                    'Segmenting: ' + str(round(task.segment_time,2)) + 's ' +
                    'Planning: ' + str(round(task.plan_grasp_time,2)) + 's ' +
                    'Grasping: ' + str(round(task.grasp_time,2)) + 's ' +
                    'Dropping: ' + str(round(task.drop_time,2)) + 's ' +
                    'Go to start: ' + str(round(task.go_to_start_time,2)) + 's '
                    )
        file.close()
        task.init_time = 0
        task.read_images_time = 0
        task.segment_time = 0
        task.plan_grasp_time = 0
        task.grasp_time = 0
        task.drop_time = 0
        task.go_to_start_time = 0

        #task.declutter()
    #    number_failed = 3
    #    del task
    print('No objects in sight, surface decluttered.')
