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




img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')


def plot_grasp(grasp, width=None, color='r', arrow_len=4, arrow_head_len=2, arrow_head_width=3,
          arrow_width=1, jaw_len=3, jaw_width=1.0,
          grasp_center_size=1, grasp_center_thickness=2.5,
          grasp_center_style='+', grasp_axis_width=1,
          grasp_axis_style='--', line_width=1.0, show_center=True, show_axis=False, scale=1.0):
    """
    Plots a 2D grasp with arrow and jaw style using matplotlib

    Parameters
    ----------
    grasp : :obj:`Grasp2D`
        2D grasp to plot
    width : float
        width, in pixels, of the grasp (overrides Grasp2D.width_px)
    color : :obj:`str`
        color of plotted grasp
    arrow_len : float
        length of arrow body
    arrow_head_len : float
        length of arrow head
    arrow_head_width : float
        width of arrow head
    arrow_width : float
        width of arrow body
    jaw_len : float
        length of jaw line
    jaw_width : float
        line width of jaw line
    grasp_center_thickness : float
        thickness of grasp center
    grasp_center_style : :obj:`str`
        style of center of grasp
    grasp_axis_width : float
        line width of grasp axis
    grasp_axis_style : :obj:`str`
        style of grasp axis line
    show_center : bool
        whether or not to plot the grasp center
    show_axis : bool
        whether or not to plot the grasp axis
    """
    # set vars for suction
    skip_jaws = False
    if not hasattr(grasp, 'width'):
        grasp_center_style = '.'
        grasp_center_size = 25
        plt.scatter(grasp.center.x, grasp.center.y, c=color, marker=grasp_center_style, s=scale * grasp_center_size)
        return

    # plot grasp center
    if show_center:
        plt.plot(grasp.center.x, grasp.center.y, c=color, marker=grasp_center_style, mew=scale * grasp_center_thickness,
                 ms=scale * grasp_center_size)
    if skip_jaws:
        return

    # compute axis and jaw locations
    axis = grasp.axis
    width_px = width
    if width_px is None and hasattr(grasp, 'width_px'):
        width_px = grasp.width_px
    g1 = grasp.center.data - (float(width_px) / 2) * axis
    g2 = grasp.center.data + (float(width_px) / 2) * axis
    g1p = g1 - scale * arrow_len * axis  # start location of grasp jaw 1
    g2p = g2 + scale * arrow_len * axis  # start location of grasp jaw 2

    # plot grasp axis
    if show_axis:
        plt.plot([g1[0], g2[0]], [g1[1], g2[1]], color=color, linewidth=scale * grasp_axis_width,
                 linestyle=grasp_axis_style)

    # direction of jaw line
    jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])

    # length of arrow
    alpha = scale * (arrow_len - arrow_head_len)

    # plot first jaw
    g1_line = np.c_[g1p, g1 - scale * arrow_head_len * axis].T
    plt.arrow(g1p[0], g1p[1], alpha * axis[0], alpha * axis[1], width=scale * arrow_width,
              head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
    jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T

    plt.plot(jaw_line1[:, 0], jaw_line1[:, 1], linewidth=scale * jaw_width, c=color)

    # plot second jaw
    g2_line = np.c_[g2p, g2 + scale * arrow_head_len * axis].T
    plt.arrow(g2p[0], g2p[1], -alpha * axis[0], -alpha * axis[1], width=scale * arrow_width,
              head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
    jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
    plt.plot(jaw_line2[:, 0], jaw_line2[:, 1], linewidth=scale * jaw_width, c=color)


class SurfaceDeclutter():

    def __init__(self):
        init_start = time.time()
        """
        Class that runs surface decluttering task
        """
        self.robot = Robot_Interface()
        self.ra = Robot_Actions(self.robot)

        # model_path = 'main/model/object_recognition_100_objects_inference_graph.pb'
        # model_path = 'main/model/real_on_real/output_inference_graph.pb'
        # model_path = '/nfs/diskstation/ajaytanwani/trained_models/real_on_real_hsr_100_objects/output_inference_graph.pb'
        # model_path = '/nfs/diskstation/ajaytanwani/trained_models/sim_then_labeled_dann_on_real_fpn_100_objects/output_inference_graph.pb'

        # model_path = '/nfs/diskstation/ajaytanwani/trained_models/real_on_real_fpn_0.7_100_objects/output_inference_graph.pb'

        # model_path = '/nfs/diskstation/ajaytanwani/trained_models/dior_adda_100_objects/output_inference_graph.pb'
        model_path = '/nfs/diskstation/ajaytanwani/trained_models/dior_dann_100_objects/output_inference_graph.pb'

        label_map_path = 'main/model/real_on_real/object-detection.pbtxt'
        self.det = Detector(model_path, label_map_path)
        self.cam = RGBD()
        self.tfBuffer = tf2_ros.Buffer()
        init_end = time.time()
        print('Initialization took %.2f seconds' %(init_end-init_start))

    def run_grasp_gqcnn(self, c_img, d_img, number_failed, object_label):
        plan_start = time.time()
        from autolab_core import RigidTransform, YamlConfig, Logger
        from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage

        from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction
        from gqcnn.utils import GripperMode, NoValidGraspsException
        from visualization import Visualizer2D as vis

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

        # import ipdb; ipdb.set_trace()
        # query policy
        policy_start = time.time()
        action = policy(state)
        if action is None:
            return number_failed + 1
        # grasp_center[x,y] in image frame
        grasp_center = [action.grasp.center[0], action.grasp.center[1]]
        grasp_angle = action.grasp.angle
        grasp_depth_m = action.grasp.depth
        grasp_height_offset = action.grasp.height_offset
        grasp_width = action.grasp.width

        # ignore corrupted depth images
        if 0.7 < grasp_depth_m < 1.55:
            pass
        else:
            print('invalid depth image')
            return number_failed + 1

        # TARGET_DIR = '/home/benno/experiments/hsr/gqcnn/10_15_elevation_v2/'
        TARGET_DIR = '/home/ajaytanwani/PycharmProjects/siemens_challenge/grasp_data/'
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

        plan_end = time.time()
        print('Planning took %.2f seconds' %(plan_end - plan_start))
        # vis final grasp
        #policy_config['vis']['final_grasp'] = False
        if policy_config['vis']['final_grasp'] and False:
            vis.figure(size=(40,40))
            vis.subplot(1,2,1)
            vis.imshow(rgbd_im.depth, vmin=policy_config['vis']['vmin'], vmax=policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=4.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))

            vis.subplot(1,2,2)
            vis.imshow(rgbd_im.color)
            vis.grasp(action.grasp, scale=4.5, show_center=True, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.savefig(TARGET_DIR + 'final_grasp_' + str(new_file_number).zfill(3))
            vis.show()

        # execute planned grasp with hsr interface
        #self.execute_gqcnn(grasp_center, grasp_angle, d_img*1000)


        # execute 2DOF grasp
        # self.execute_gqcnn_2DOF(grasp_center, grasp_depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img*1000, object_label)
        # return 0
        return action


    def focus_on_target_zone(self, d_img, bbox):
        d_img[:, :bbox.xmin] = 0
        d_img[:, bbox.xmax:] = 0
        d_img[:bbox.ymin, :] = 0
        d_img[bbox.ymax:, :] = 0
        # plt.imshow(d_img)
        # plt.show()
        return d_img

    def execute_gqcnn(self, grasp_center, grasp_angle, depth_image_mm):
        grasp_direction = np.array([math.sin(grasp_angle), math.cos(grasp_angle)])
        grasp_direction_normalized = grasp_direction / np.linalg.norm(grasp_direction)
        self.ra.execute_grasp(grasp_center, grasp_direction_normalized, depth_image_mm, 0, 500.0)

    def execute_gqcnn_2DOF(self, grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img, object_label):
        self.ra.execute_2DOF_grasp(grasp_center, depth_m, grasp_angle, grasp_width, grasp_height_offset, d_img, object_label)

    def get_bboxes_from_net(self, img, sess=None):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_dict, vis_util_image = self.det.predict(rgb_image, sess=sess)
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

    def declutter(self, number_failed):
        self.ra.go_to_start_pose()
        time.sleep(1)
        c_img, d_img = self.read_RGBD_image()
        # convert depth image from mm to m because dexnet uses m
        depth_image_mm = np.asarray(d_img[:,:])
        depth_image_m = depth_image_mm/1000

        number_failed = self.run_grasp_gqcnn(c_img, depth_image_m, number_failed)
        return number_failed
        
    
    def segment(self, c_img_filename, d_img_filename):

        c_img = cv2.imread(c_img_filename)

        d_img = np.load(d_img_filename)


        # self.ra.go_to_start_pose()
        # time.sleep(2)
        # c_img, d_img = self.read_RGBD_image()
        # plt.imshow(c_img)
        # plt.show()
        # return 3
        with self.det.detection_graph.as_default():
            with tensorflow.Session() as sess:
                main_mask = crop_img(c_img, simple=True, arc=False)
                col_img = ColorImage(c_img)
                workspace_img = col_img.mask_binary(main_mask)

                bboxes, vis_util_image = self.get_bboxes_from_net(c_img, sess=sess)
                bboxes = [bbox for bbox in bboxes if bbox.prob > 0.6]

                vis_util_image_rgb = cv2.cvtColor(vis_util_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(c_img_filename + "_obj_det" + ".png", vis_util_image_rgb)

                actions_list = []
                for target_id, target_bbox in enumerate(bboxes):
                    d_img_mask = self.focus_on_target_zone(deepcopy(d_img), target_bbox)
                    depth_image_m = np.asarray(d_img_mask[:,:])
                    # depth_image_m = depth_image_mm/1000
                    # import ipdb; ipdb.set_trace()
                    action = self.run_grasp_gqcnn(vis_util_image_rgb, depth_image_m, 0, target_bbox.label)
                    actions_list.append(action)
                    # import ipdb; ipdb.set_trace()
                    print (target_id,target_bbox.label)
                    # if target_id == 2:
                    #     break


                # vis.grasp(action.grasp, scale=4.5, show_center=False, show_axis=True)
                # vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
                #
                # vis.subplot(1, 2, 2)
                # vis.imshow(rgbd_im.color)
                # vis.grasp(action.grasp, scale=4.5, show_center=True, show_axis=True)
                # vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
                # vis.savefig(TARGET_DIR + 'final_grasp_' + str(new_file_number).zfill(3))
                # vis.show()


                # import ipdb; ipdb.set_trace()
                TARGET_DIR = '/home/ajaytanwani/PycharmProjects/siemens_challenge/'
                plt.figure(figsize=(5,5))
                # plt.imshow(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB))
                plt.imshow(vis_util_image)

                for action_id in range(len(actions_list)):
                    # import ipdb; ipdb.set_trace()
                    if actions_list[action_id] is not 1:
                        plot_grasp(actions_list[action_id].grasp, scale=2.5, show_center=True, show_axis=True, arrow_len=6, arrow_head_len=4, arrow_head_width=1, arrow_width=1, jaw_len=4.0, jaw_width=1.0)
                    # vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(act.grasp.depth, act.q_value))
                    plt.hold(True)
                # vis.hold(True)
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                # plt.savefig(TARGET_DIR + c_img_filename + '_final_grasp.png', dpi=100, bbox_inches='tight', pad_inches=0.0)
                plt.savefig(TARGET_DIR + c_img_filename.split('/')[1] + '_final_grasp.eps', format='eps', dpi=300)

                import ipdb;
                ipdb.set_trace()

                plt.show()

                    # return number_failed

                # if len(bboxes) == 0:
                #     print("Cleared the workspace")
                #     print("Add more objects, then resume")
                #     return 3
                # else:
                #     y_max = 0
                #     for bbox in bboxes:
                #         if bbox.ymax >= y_max:
                #             y_max = bbox.ymax
                #             target_bbox = bbox
                #     max_prob = target_bbox.prob
                #     for bbox in bboxes:
                #         if bbox.ymax > y_max - 50 and bbox.prob > max_prob:
                #             max_prob = bbox.prob
                #             target_bbox = bbox
                #     # rgb_image = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                #     # rgb_image = target_bbox.draw(rgb_image)
                #     # plt.imshow(rgb_image)
                #     # plt.show()
                #     vis_util_image_rgb = cv2.cvtColor(vis_util_image, cv2.COLOR_BGR2RGB)
                #     cv2.imwrite(c_img_filename + "_obj_det" +".png", vis_util_image_rgb)
                #     # return 3
                #     d_img = self.focus_on_target_zone(d_img, target_bbox)
                #     depth_image_m = np.asarray(d_img[:,:])
                #     # depth_image_m = depth_image_mm/1000
                #     import ipdb; ipdb.set_trace()
                #     number_failed = self.run_grasp_gqcnn(vis_util_image_rgb, depth_image_m, 0, target_bbox.label)
                #     return number_failed


if __name__ == "__main__":

    # plt.figure(figsize=(10, 10))
    # plt.imshow(inference_result['depth_image'], vmin=0.6, vmax=0.9)
    # plot_grasp(inference_result, scale=2.5, show_center=False, show_axis=True)
    # plt.title('Planned grasp on depth (Q=%.3f)' % (inference_result['q_value']))


    # for iter in xrange(3,4):
    #     depth_im_filename = os.path.join('hsr_test/depth_raw_' + str(iter).zfill(4) + '.npy')
    #     rgb_im_filename = os.path.join('hsr_test/rgb_raw_' + str(iter).zfill(4) + '.png')



    image_ids=[186,147]
    for iter in xrange(2):
        depth_im_filename = os.path.join('hsrsimtest/depth_' + str(image_ids[iter]) + '.npy')
        rgb_im_filename = os.path.join('hsrsimtest/rgb_' + str(image_ids[iter]) + '.png')

        task = SurfaceDeclutter()

        number_failed = task.segment(rgb_im_filename, depth_im_filename)

        del task


    # while number_failed <= 2:
    #     print('Starting new run with %d fails in a row now' %(number_failed))
    #     task = SurfaceDeclutter()
    #     # rospy.spin()
    #     number_failed = task.segment(number_failed)
    #     #number_failed = task.declutter(number_failed)
    # #    number_failed = 3
    #     del task
    # print('No objects in sight, surface decluttered.')
