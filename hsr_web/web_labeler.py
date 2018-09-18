
import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2
import numpy as np

sharer = Pyro4.Proxy("PYRONAME:shared.server")

#robot interface
# GLOBAL_PATH = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/"
CANVAS_DIM = 420.0

class Web_Labeler:

	def __init__(self, num_robots, robot_id = 0):
		self.count = 0
		self.robot_id = robot_id
		self.num_robots = num_robots

	def label(self, img_path):
	    global sharer

	    img = cv2.imread(img_path)

	    h_,w_,dim = img.shape

	    sharer.set_img(img_path)
	    sharer.set_img_ready(True)

	    print("Web Labeler waiting")
	    while not sharer.is_labeled():
	        pass
	    print("Web Labeler done")

	    label = sharer.get_label_data()
	    sharer.set_labeled(False)

	    # self.count += 1
	    return self.rescale_labels(label,w_,h_)

	def rescale_labels(self,labels,w_,h_):

		if(labels == None):
			return

		for label in labels['objects']:

			non_scaled = label['coords']

			x_min = non_scaled[0]*(w_/CANVAS_DIM)
			y_min = non_scaled[1]*(h_/CANVAS_DIM)

			x_max = non_scaled[2]*(w_/CANVAS_DIM)
			y_max = non_scaled[3]*(h_/CANVAS_DIM)

			label['coords'] = [int(a) for a in [x_min, y_min, x_max,y_max]]

		return labels

	def send_confidences(self, confidences_path):
		global sharer

		sharer.set_confidences(confidences_path)

	def label_image(self, img_path):

		robots = np.arange(self.num_robots)
		np.random.shuffle(robots)

		for robot in robots:

			print "Desired Robot: " + str(self.robot_id)
			print "Queued Robot: " + str(robot)
			frame = os.getcwd() + "/" + img_path
			# import ipdb; ipdb.set_trace()
			label_data = self.label(frame)
			print("Label Data: ",label_data)
			print("Time:" , float(label_data['time'])/1000.0)
			print("Latency: ", float(label_data['latency'])/1000.0)
			time.sleep(5)
			# pickle.dump(label_data, open("data/labels/" + str(robot) + ".p", 'wb'))
			if robot == self.robot_id:
				print "ROBOT LABELED"
				return label_data
				break;

'''
Example script below
'''
# labeler = Web_Labeler(5)
# img_path = "data/images/frame_0.png"
# labeler.label_image(img_path)

# for i in range(2):
# 	img_path = "data/images/frame_" + str(i) + ".png"
# 	conf_path = "data/confidences/frame_" + str(i) + ".png"
# 	labeler.send_confidences(conf_path)
# 	label_data = labeler.label_image(img_path)
# 	print(label_data)
# 	print(float(label_data['time'])/1000.0)
# 	print(float(label_data['latency'])/1000.0)
