import os
import numpy as np

"""CONFIG FILE FOR FOG ROBOTICS DECLUTTERING PROJECT"""

"""OPTIONS FOR DEMO"""
#whether to save rollouts
COLLECT_DATA = False

#whether to show plots/ask for success
QUERY = False

#whether to attempt multiple grasps from just 1 image; if true, susceptible to error from open loop control
CHAIN_GRASPS = False



"""TABLE SETUP SPECIFIC VALUES"""

labels = ["Screwdriver", "Scrap", "Tube", "Tape"] #ensure this matches server.py in hsr_web
# HUES_TO_BINS = {"cyan": 0 , "blue": 1, "green-yellow": 2, "green": 3, "yellow": 4, "red": 5, "orange": 6, "black": 7}
HUES_TO_BINS = {'yellow':0, 'cyan': 1, 'red': 2}

net_labels = {1: "Tube", 2: "Scrap", 3: "Screwdriver", 4: "Tape"}
CONFIDENCE_THRESH = 0.3
EVALUATE = False
ISOLATED_TOL = 100

ASKING_FOR_HELP_POLICY = "NO_HELP" #Options: "NO HELP", "SIMPLE", "MODEL_BASED"
NUM_ROBOTS_ON_NETWORK = 1

"""EMPIRICALLY TUNED PARAMETERs"""
#CONENCTED COMPONENTS ALG PARAMETERS
#number of pixels apart to be singulated
DIST_TOL = 5
#background range for thresholding the image
COLOR_TOL = 34
#number of pixels necssary for a cluster
SIZE_TOL = 50
#amount to scale image down by to run algorithm (for speed)
SCALE_FACTOR = 2

#ROBOT PARAMETERS
#distance grasp extends
LINE_SIZE = 38
#side length of square that checks grasp collisions
#increase range to reduce false positives
CHECK_RANGE = 2

#HSV PARAMETERS
#cv2 range for HSV hue values
HUE_RANGE = 180.0
#cv2 range for HSV sat values
SAT_RANGE = 255.0
#cv2 range for HSV value values
VALUE_RANGE = 255.0
#fraction of saturation range that is white
WHITE_FACTOR = 0.15 #0.1
#fraction of value range that is black
BLACK_FACTOR = 0.3
#carving up HSV color space by lego-specific colors
#see https://en.wikipedia.org/wiki/HSL_and_HSV (scaled down from 360 to 180 degrees)
HUE_VALUES = {90: "cyan", 120: "blue", 0: "red", 10: "orange", 30: "yellow",
	60: "green", 35: "green-yellow"}
#include black as special case
ALL_HUE_VALUES = HUE_VALUES.copy()
ALL_HUE_VALUES[-1] = "black"

#singulation parameters
#factor to move start point by so it is not in the pile
SINGULATE_START_FACTOR = 1.2
#factor to move end point by towards start point
SINGULATE_END_FACTOR = 0.75

"""PATHS AND DATASET PARAMETERS"""
robot_name = "hsr"
# robot_name = "fetch"
# robot_name = None

IMG_MODULE = "tpc.perception.image"

#convenience parameter to change paths based on machine
computer = "michael"
if computer == "michael":
	ROOT_DIR = '/media/autolab/1tb/data/'
	DATA_PATH = ROOT_DIR + 'tpc/'
	IMG_MODULE = 'tpc.perception.image'
	WEB_PATH = '/home/autolab/Workspaces/michael_working/hsr_web'
	SIEMENS_PATH = '/home/autolab/Workspaces/michael_working/siemens_challenge'
elif computer == "chris":
	ROOT_DIR = '/Users/chrispowers/Documents/research/tpc/'
	DATA_PATH = ROOT_DIR + 'data/'
	IMG_MODULE = 'tpc.perception.image'
	WEB_PATH = '/Users/chrispowers/Documents/research/hsr_web'
elif computer == "zisu":
	WEB_PATH = '/home/zisu/simulator/hsr_web'
	SIEMENS_PATH = '/home/zisu/simulator/siemens_challenge/'
ROLLOUT_PATH = DATA_PATH+'rollouts-3-9/'
