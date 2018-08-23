import xml.etree.ElementTree as et
import numpy as np
import trimesh
import os
import math

SLICE_UPPERBOUND = 0.008
MIN_EDGE_UPPERBOUND = 0.08
MAX_EDGE_UPPERBOUND = 0.3

SLICE_LOWERBOUND = 0.003
# MIN_EDGE_LOWERBOUND = 0.005
RIDGE_LOWERBOUND = 0.3


SIZES = np.arange(0.8, 1.3, 0.1)

def parse_box(filename, scale, new_model_name):
	template = et.parse('model_box_template.sdf')
	config_template = et.parse("model_template.config")

	root = template.getroot()
	cfg_root = config_template.getroot()

	mesh_name = filename.split(".")[0]
	collision_name = filename.split("_")[0]
	root[0].set('name', new_model_name)
	cfg_root[0].text = new_model_name
	print(collision_name)

	mesh = trimesh.load_mesh(collision_name+".obj")

	visual = root[0][0][5]
	collision = root[0][0][6]

	visual[0][0][0].text = os.path.abspath(mesh_name+".dae")

	# # TODO: figure out scale
	
	raw_box = np.max(np.diff(mesh.bounding_box.vertices, axis=0), axis=0)

	x_size, y_size, z_size = raw_box
	while scale ** 2 * np.min(raw_box) * np.median(raw_box) < SLICE_LOWERBOUND or scale * math.sqrt(x_size**2+y_size**2+z_size**2) < RIDGE_LOWERBOUND:
		print(scale)
		scale += 0.1
	while scale * np.min(raw_box) > MIN_EDGE_UPPERBOUND or scale ** 2 * np.min(raw_box) * np.median(raw_box) > SLICE_UPPERBOUND or scale * np.max(raw_box) > MAX_EDGE_UPPERBOUND:
		scale -= 0.1


	mass = root[0][0][1][0]
	mass.text = str(0.5)

	ixx = root[0][0][1][1][0]
	iyy = root[0][0][1][1][3]
	izz = root[0][0][1][1][5]
	ixx.text = str(0.5/12*(y_size**2+z_size**2))
	iyy.text = str(0.5/12*(x_size**2+z_size**2))
	izz.text = str(0.5/12*(x_size**2+y_size**2))

	visual[0][0][1].text = str(scale)+" "+str(scale)+" "+str(scale)

	collision[3][0][0].text = str(x_size*scale)+" "+str(y_size*scale)+" "+str(z_size*scale)
	# collision[3][0][0].text = str(x_size)+" "+str(y_size)+" "+str(z_size)

	x, y, z = mesh.bounding_box.centroid
	
	pose = root[0][0][1][2]
	pose.text = str(x*scale)+" "+str(-z*scale)+" "+str(y*scale)+" "+"1.57"+" "+"0"+" "+"3.14"
	# pose.text = str(x)+" "+str(-z)+" "+str(y)+" "+"1.57"+" "+"0"+" "+"3.14"

	collision[2].text = str(x*scale)+" "+str(-z*scale)+" "+str(y*scale)+" "+"1.57"+" "+"0"+" "+"3.14"
	# collision[2].text = str(x)+" "+str(-z)+" "+str(y)+" "+"1.57"+" "+"0"+" "+"3.14"

	if not os.path.exists(new_model_name):
		os.makedirs(new_model_name)
	template.write(new_model_name+"/"+"model.sdf")
	config_template.write(new_model_name+"/"+"model.config")

if __name__ == '__main__':
	for item in os.listdir("."):
		if (len(item.split(".")) == 2 and item.split(".")[1] == "dae"):
			print(item)
			for i in range(5):
				if not os.path.exists(item.split(".")[0]+"_"+str(i)):
					parse_box(item, SIZES[i], item.split(".")[0]+"_"+str(i))
