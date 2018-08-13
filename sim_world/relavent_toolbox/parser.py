import xml.etree.ElementTree as et
import numpy as np
import trimesh
import os

BOX_UPPERBOUND = 0.2


def parse_box(filename):
	template = et.parse('model_box_template.sdf')
	config_template = et.parse("model_template.config")

	root = template.getroot()
	cfg_root = config_template.getroot()

	new_model_name = filename.split(".")[0]
	root[0].set('name', new_model_name)
	cfg_root[0].text = new_model_name

	mesh = trimesh.load_mesh(new_model_name+".obj")

	visual = root[0][0][5]
	collision = root[0][0][6]

	visual[0][0][0].text = os.path.abspath(new_model_name+".dae")

	# # TODO: figure out scale
	
	raw_box = np.max(np.diff(mesh.bounding_box.vertices, axis=0), axis=0)
	# scale = BOX_UPPERBOUND / max(raw_box)
	x_size, y_size, z_size = raw_box

	mass = root[0][0][1][0]
	mass.text = str(0.5)

	ixx = root[0][0][1][1][0]
	iyy = root[0][0][1][1][3]
	izz = root[0][0][1][1][5]
	ixx.text = str(0.5/12*(y_size**2+z_size**2))
	iyy.text = str(0.5/12*(x_size**2+z_size**2))
	izz.text = str(0.5/12*(x_size**2+y_size**2))

	# visual[0][0][1].text = str(scale)+" "+str(scale)+" "+str(scale)

	collision[3][0][0].text = str(x_size)+" "+str(y_size)+" "+str(z_size)
	# collision[3][0][0].text = str(y_size)+" "+str(x_size)+" "+str(z_size)

	x, y, z = mesh.bounding_box.centroid
	
	pose = root[0][0][1][2]
	pose.text = str(x)+" "+str(-z)+" "+str(y)+" "+"1.57"+" "+"0"+" "+"3.14"


	collision[2].text = str(x)+" "+str(-z)+" "+str(y)+" "+"1.57"+" "+"0"+" "+"3.14"
	# collision[2].text = str(y)+" "+str(x)+" "+str(z)+" "+"0"+" "+"0"+" "+"0"

	if not os.path.exists(new_model_name):
		os.makedirs(new_model_name)
	template.write(new_model_name+"/"+"model.sdf")
	config_template.write(new_model_name+"/"+"model.config")

if __name__ == '__main__':
	for item in os.listdir("."):
		if (len(item.split(".")) == 2 and item.split(".")[1] == "obj"):
			print(item)
			parse_box(item)