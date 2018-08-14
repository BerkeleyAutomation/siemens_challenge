import cv2
import IPython
import time
import numpy as np
import sys

from tpc.manipulation.robot_actions import Robot_Actions
# from tpc.data_logger import DataLogger
import tpc.config.config_tpc as cfg

import thread
import tf
import rospy
import os
import json
import shutil

if cfg.robot_name == "hsr":
    from core.hsr_robot_interface import Robot_Interface
elif cfg.robot_name == "fetch":
    from core.fetch_robot_interface import Robot_Interface 
elif cfg.robot_name is None:
    from tpc.offline.robot_interface import Robot_Interface

from spawn_object_script import *
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetWorldProperties, SetPhysicsProperties
import gazebo_msgs.msg
from std_msgs.msg import Float64

from segmentation import *
# from segmentation_rgb import *

def depth_scaled_to_255(img):
	img = 255.0/np.max(img)*img
	img = np.array(img,dtype=np.uint8)
	
	img[:,:] = cv2.equalizeHist(img[:,:])
	# cv2.imshow('debug.png',img)
	return img



class DataCollection():

	def __init__(self):
		"""
		Class that runs decluttering task

		"""
		self.robot = Robot_Interface()
		self.ra = Robot_Actions(self.robot)

		self.dm, self.sm, self.om = setup_delete_spawn_service()

		print "Finished init"

		clean_floor(self.dm, self.om)

		time.sleep(2)


	def collect(self, dataset_size=100, start=0):

		IMDIR = 'sim_data/dataset_08_013_2018/'
		if not os.path.exists(IMDIR):
			os.makedirs(IMDIR)
		# IMDIR = 'sim_data/test/'

		i = 0

		while i < dataset_size:
			print(i+start)
			
			if not os.path.exists(IMDIR+str(i+start)):
				os.makedirs(IMDIR+str(i+start))

			n = np.random.randint(5, 10)

			spawn_from_uniform(n, self.sm)
			
			labels = get_object_list(self.om)
			time.sleep(0.1)
			self.ra.go_to_start_position()
			self.ra.go_to_start_pose()
			
			time.sleep(0.1)

			for j in range(len(labels)):
				
				c_img, d_img = self.robot.get_img_data()
				delete_object(labels[len(labels) - 1 - j], self.dm)
				cv2.imwrite(IMDIR+str(i+start)+'/rgb_{}.png'.format(str(len(labels) -1- j)), c_img)
				cv2.imwrite(IMDIR+str(i+start)+'/depth_{}.png'.format(str(len(labels) -1- j)), depth_scaled_to_255(np.array((d_img * 1000).astype(np.int16))))
				if j == 0:
					cv2.imwrite(IMDIR+'/rgb_{}.png'.format(str(i+start)), c_img)
					cv2.imwrite(IMDIR+'/depth_{}.png'.format(str(i+start)), depth_scaled_to_255(np.array((d_img * 1000).astype(np.int16))))

			clean_floor(self.dm, self.om)
			time.sleep(0.1)

			c_img, d_img = self.robot.get_img_data()

			cv2.imwrite(IMDIR+str(i+start)+'/rgb_background.png', c_img)
			cv2.imwrite(IMDIR+str(i+start)+'/depth_background.png', depth_scaled_to_255(np.array((d_img * 1000).astype(np.int16))))
			# print(np.array((d_img * 1000).astype(np.int16)).dtype)

			
			with open(IMDIR+str(i+start)+"/labels.json", 'w') as f:
				json.dump(labels, f)

			find_item_masks(IMDIR+str(i+start))
			compare, diff, seg_image, bb_image = draw_masks(IMDIR+str(i+start))



			if diff > 640*480/300:
				os.remove(IMDIR+'/rgb_{}.png'.format(str(i+start)))
				os.remove(IMDIR+'/depth_{}.png'.format(str(i+start)))
			else:
				cv2.imwrite(IMDIR+'/compare_{}.png'.format(str(i+start)), compare)
				cv2.imwrite(IMDIR+'/seg_{}.png'.format(str(i+start)), seg_image)
				cv2.imwrite(IMDIR+'/bb_{}.png'.format(str(i+start)), bb_image)
				create_segment_label(IMDIR, str(i+start))

			shutil.rmtree(IMDIR+str(i+start))
			time.sleep(0.5)

			

			if diff <= 640*480/300:
				i+=1
			# i += 1

			

			# time.sleep(3)
			# self.ra.go_to_start_position()
			# self.ra.go_to_start_pose()
			
			# time.sleep(2)



if __name__ == "__main__":
	DataCollection().collect(dataset_size=10000, start=0)




