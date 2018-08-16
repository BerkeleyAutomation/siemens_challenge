import bpy
import os

# if __name__ == '__main__':
for item in os.listdir("sim_world/toolbox/"):
	if (len(item.split(".")) == 2 and item.split(".")[1] == "obj"):
		name = item.split(".")[0]
		bpy.ops.import_scene.obj(filepath="sim_world/toolbox/"+item)
		bpy.ops.wm.collada_export(filepath="sim_world/toolbox/"+name+".dae", open_sim=True)
		for o in bpy.data.objects:
			if o.type =="MESH":
				o.select = True
			else:
				o.select = False
		bpy.ops.object.delete()

for item in os.listdir("sim_world/toolbox/"):
	if (len(item.split(".")) == 2 and item.split(".")[1] == "dae"):
		bpy.ops.wm.collada_import(filepath="sim_world/toolbox/"+item)
		bpy.ops.wm.collada_export(filepath="sim_world/toolbox/"+item, open_sim=True)
		for o in bpy.data.objects:
			if o.type =="MESH":
				o.select = True
			else:
				o.select = False
		bpy.ops.object.delete()
