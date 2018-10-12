import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys

from tpc.perception.cluster_registration import run_connected_components, display_grasps
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.manipulation.robot_actions import Robot_Actions
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance
from tpc.helper import Helper
# from tpc.data_logger import DataLogger
import tpc.config.config_tpc as cfg
import importlib

import thread
import tf
import rospy
import json

if cfg.robot_name == "hsr":
    from core.hsr_robot_interface import Robot_Interface
elif cfg.robot_name == "fetch":
    from fetch_core.robot_interface import Robot_Interface
elif cfg.robot_name is None:
    from tpc.offline.robot_interface import Robot_Interface

sys.path.append('/home/zisu/simulator/hsr_web')
from web_labeler import Web_Labeler

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

from detection import Detector
sys.path.append('/home/zisu/simulator/siemens_challenge/sim_world')
from spawn_object_script import *
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetWorldProperties

"""
This class is for use with the robot
Pipeline it tests is labeling all objects in an image using web interface,
then choosing between and predicting a singulation or a grasp,
then the robot executing the predicted motion
"""
Categories = [(1, -0.3, 0), (1, -0.1, 0), (1, 0.1, 0), (1, 0.3, 0)]

def broadcast_class(i, br):
    a = tf.transformations.quaternion_from_euler(ai=0.0,aj=-3.14,ak=1.57)
    # b = tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=1.57)

    # base_rot = tf.transformations.quaternion_multiply(a,b)

    while True:
        br.sendTransform(Categories[i],
            #tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
            a,
            rospy.Time.now(),
            'ar_marker/'+str(i+8),
            #'head_rgbd_sensor_link')
            'map')

def loop_broadcast_classes():
    for i in range(4):
        br = tf.TransformBroadcaster()
        thread.start_new_thread(broadcast_class,(i,br))

class SiemensDemo():

    def __init__(self, auto=False):
        """
        Class that runs decluttering task

        """
        self.robot = Robot_Interface()
        self.helper = Helper(cfg)
        self.ra = Robot_Actions(self.robot)
        # self.dl = DataLogger("stats_data/model_base", cfg.EVALUATE)
        self.web = Web_Labeler(cfg.NUM_ROBOTS_ON_NETWORK)

        model_path = 'main/output_inference_graph.pb'
        label_map_path = 'main/object-detection.pbtxt'
        self.det = Detector(model_path, label_map_path)

        self.auto = auto
        if auto:
            self.dm, self.sm, self.om = setup_delete_spawn_service()
            clean_floor(self.dm, self.om)
            spawn_from_uniform(6, self.sm)

        print "Finished init"

    def run_grasp(self, bbox, c_img, col_img, workspace_img, d_img):
        print("Grasping a " + cfg.labels[bbox.label])
        try:
            group = bbox.to_group(c_img, col_img)
        except ValueError:
            return 
            
        display_grasps(workspace_img, [group])

        self.ra.execute_grasp(group.cm, group.dir, d_img, class_num=bbox.label)

    def run_singulate(self, singulator, d_img):
        print("singulating")
        waypoints, rot, free_pix = singulator.get_singulation()
        singulator.display_singulation()
        self.ra.execute_singulate(waypoints, rot, d_img)

    def get_bboxes_from_net(self, path):
        output_dict, vis_util_image = self.det.predict(path)
        plt.savefig('debug_imgs/predictions.png')
        plt.close()
        plt.clf()
        plt.cla()
        img = cv2.imread(path)
        boxes = format_net_bboxes(output_dict, img.shape)
        return boxes, vis_util_image

    # def get_bboxes_from_web(self, path):
    #     labels = self.web.label_image(path)

    #     objs = labels['objects']
    #     bboxes = []
    #     vectors = []
    #     for obj in objs:
    #         if obj['motion'] == 1:
    #             coords = obj['coords']
    #             p0 = [coords[1], coords[0]]
    #             p1 = [coords[3], coords[2]]
    #             vectors.append(([p0, p1], obj['class']))
    #         else:
    #             bboxes.append(Bbox(obj['coords'], obj['class']))
    #     return bboxes, vectors

    def determine_to_ask_for_help(self,bboxes,col_img):
        bboxes = deepcopy(bboxes)
        col_img = deepcopy(col_img)

        single_objs = find_isolated_objects_by_overlap(bboxes)

        if len(single_objs) > 0:
            return False
        else:
            single_objs = find_isolated_objects_by_distance(bboxes, col_img)
            return len(single_objs) == 0

    def get_bboxes(self, path,col_img):
        boxes, vis_util_image = self.get_bboxes_from_net(path)
        vectors = []

        #low confidence or no objects
        # if len(boxes) == 0 or self.determine_to_ask_for_help(boxes,col_img):
        #     self.helper.asked = True
        #     self.helper.start_timer()
        #     boxes, vectors = self.get_bboxes_from_web(path)
        #     self.helper.stop_timer()
            # self.dl.save_stat("duration", self.helper.duration)
            # self.dl.save_stat("num_online", cfg.NUM_ROBOTS_ON_NETWORK)

        return boxes, vectors, vis_util_image

    def siemens_demo(self):
        print("inside siemens demo!")
        loop_broadcast_classes()
        
        self.ra.go_to_start_pose()
        c_img, d_img = self.robot.get_img_data()

        summary = {}



        i = 0
        while not (c_img is None or d_img is None):
            path = "/home/zisu/simulator/siemens_challenge/debug_imgs/web.png"
            cv2.imwrite(path, c_img)
            # time.sleep(0.1) #make sure new image is written before being read

            # print "\n new iteration"
            # start timing 
            start = time.time()
            print "start iter: " + str(i)

            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            bboxes, vectors, vis_util_image = self.get_bboxes(path,col_img)

            if len(bboxes) > 0:
                box_viz = draw_boxes(bboxes, c_img)
                cv2.imwrite("debug_imgs/box.png", box_viz)
                single_objs = find_isolated_objects_by_overlap(bboxes)
                print("single_objs: {}".format(single_objs))
                grasp_success = 1.0
                if len(single_objs) == 0:
                    clean_floor(self.dm, self.om)
                    spawn_from_uniform(6, self.sm)
                    print("Cleared the workspace")
                    print("Add more objects, then resume")

                if len(single_objs) > 0:
                    to_grasp = select_first_obj(single_objs)
                    singulation_time = 0.0
                    self.run_grasp(to_grasp, c_img, col_img, workspace_img, d_img)
                    # grasp_time = time.time() - start
                    # summary["time"].append(grasp_time)
                    # IPython.embed()
                    # grasp_success = self.dl.record_success("grasp", other_data=[c_img, vis_util_image, d_img])
                                
                # else:
                #     #for accurate singulation should have bboxes for all
                #     groups = [box.to_group(c_img, col_img) for box in bboxes]
                #     groups = merge_groups(groups, cfg.DIST_TOL)
                #     singulator = Singulation(col_img, main_mask, [g.mask for g in groups])
                #     self.run_singulate(singulator, d_img)
                #     sing_start = time.time()
                #     # singulation_success = self.dl.record_success("singulation", other_data=[c_img, vis_util_image, d_img])
                #     singulation_time = time.time()-sing_start

                # if cfg.EVALUATE:
                #     reward = self.helper.get_reward(grasp_success,singulation_time)
                    # self.dl.record_reward(reward)
            # elif len(vectors) > 0:
            #     waypoints, class_labels = vectors[0]
            #     rot = 0
            #     singulator = Singulation(col_img, main_mask, [], goal_p=waypoints[-1], waypoints=waypoints, gripper_angle=rot)
            #     self.run_singulate(singulator, d_img)

            i+=1
            if self.auto and i % 3 == 0:
                clean_floor(self.dm, self.om)
                spawn_from_uniform(6, self.sm)
                print("Cleared the workspace")
                print("Add more objects, then resume")

            # else:
            #     print("Cleared the workspace")
            #     print("Add more objects, then resume")
                # IPython.embed()
            print(summary)
            with open('summary.json', 'w') as outfile:
                json.dump(summary, outfile)

            if i >= 30:
                with open('summary.json', 'w') as outfile:
                    json.dump(summary, outfile)

            self.ra.go_to_start_position()
            
            self.ra.go_to_start_pose()
            c_img, d_img = self.robot.get_img_data()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-auto", action="store_true")
    args = parser.parse_args()

    task = SiemensDemo(args.auto)
    task.siemens_demo()
