from collada import *
import numpy as np

# effect2 = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))

def create_random_rgb_config():
	prob = np.random.random()
	brightness = np.random.random()

	if prob < 0.7:
		return (brightness, brightness, brightness)
	elif prob < 0.8:
		return (brightness, 0, 0)
	elif prob < 0.9:
		return (0, brightness, 0)
	else:
		return (0, 0, brightness)

def random_paint(input_filename, output_filename):
	mesh = Collada(input_filename)
	for i in range(len(mesh.scene.nodes)):
		num = len(mesh.effects)
		color = create_random_rgb_config()
		effect = material.Effect("effect" + str(num+i), [], "phong", diffuse=color, specular=color)
		mat = material.Material("material_0_"+str(num+i)+"_0ID", "material_0_"+str(num+i)+"_0", effect)
		mesh.effects.append(effect)
		mesh.materials.append(mat)
		if not mesh.scene.nodes[i].children[0].materials:
			mesh.scene.nodes[i].children[0].materials = [scene.MaterialNode("materialsref" + str(i), mat, inputs=[])]
		mesh.scene.nodes[i].children[0].materials[0].target = mat
		mesh.scene.nodes[i].children[0].materials[0].symbol = mat.name
	mesh.write(output_filename)

if __name__ == '__main__':
	if not os.path.exists("painted_meshes"):
		os.makedirs("painted_meshes")

	for item in os.listdir("."):
		if (len(item.split(".")) == 2 and item.split(".")[1] == "dae"):
			for i in range(0, 5):
				random_paint(item, "painted_meshes/"+item.split(".")[0]+"_"+str(i)+".dae")
	# for i in range(0, 5):
	# 	random_paint("screwdriver5.dae", "painted_meshes/screwdriver5_"+str(i)+".dae")