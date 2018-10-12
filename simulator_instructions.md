# Simulator usage

## Dependencies

Current working version of the simulator world/object model depends on **Gazebo 7.x** and **ros-kinetics**.

Please make sure gazebo's default material and world database is installed under path `/usr/share/gazebo-7/`, or some materials in some of the world files would fail to load.

*TODO: Change path dependent parts in code*

## Running simulation

First of all, change path for world file inside `sim_world/filename.launch` to a viable path on your computer:

For example for `office_env.launch` (Current working office environment)
```
<arg name="world_name" value="/home/zisu/simulator/siemens_challenge/sim_world/office_env.world" />
```
to 
```
<arg name="world_name" value="/usr/your_path/siemens_challenge/sim_world/office_env.world" />
```

we can run this `roslaunch` command if we have the right files:

```
roslaunch office_env.launch
```

And then you should see a world with hsr robot show up in Gazebo.

`floor_plan.launch` opens a file that mimics the lab floor setup.

None of the two world file is path dependent. 

### Creating a simulated environment

You can start by copying the existing code in 

```
test_room.world
test_room.launch
```

which gives you an empty room with walls. 

Be aware of the parameters you use for your environment: to support accurate navigation and joint control, you must include
```
name="use_laser_odom" value="true"
```
in your launch file and make sure your environment is surrounded by obscures. In our case, building environments with walls is more ideal.

### Importing object models

The object model dataset we have right now is contained under sim_world/toolbox/ .

The model sdf file we are using is also path-dependent: to reuse existing models in repo, please modify the line under `model.sdf`:

```
<uri>/home/zisu/simulator/siemens_challenge/sim_world/toolbox/tape3.obj</uri>
```
to
```
<uri>/usr/your_path/siemens_challenge/sim_world/toolbox/tape3.obj</uri>
```

A parser is also created to automatically take care of this path change for you. Check a latter part of this instruction (Running simulation/Creating object models) for more information.

#### Import object using GUI

This can be done under "insert" tag if you are using Gazebo 7.x.

#### Import object in python

Every method you need for this is contained in `sim_world/spawn_object_script.py`; please refer to the documentation and also the main function of that file for particular usage.

Note that the exact path to the target object folder to use the script: line 11 variable `MODEL_PATH` should contain the correct absolute path to the target folder with object models; line 13 `MODEL_TYPE` should contain a dictionary of key:classname and value:count. Every mesh in the path should be named as [classname]+[id(starting from 1)]+_+[painted version(starting from 0)] to allow the script run properly. 

### Creating object models

If you have both COLLADA and obj meshes for the same object, `sim_world/relavent_toolbox/parser.py` can handle auto generation of object models out of existing .dae and .obj meshes inside the same folder, assuming 1) they have the same filename 2) the two meshes and the script are in the same folder. If you want to use other object names, please change line 20 and 25 to import other .obj and .dae files respectively.

#### Paint object meshes

`sim_world/relavent_toolbox/painting_script.py` would go through all the .dae meshes in the same folder as the script and randomly paint them according to custom-specified scheme; please modify "color" field on line 23 if you want to use other schemes for generating colors. That field has to be a 3-float(0-1) tuple correspond to RGB values respectively. 

Note that the script currently iterate through all the existing group of meshes and addon colors to it; if the mesh is of only one group, all painted meshes would turn out as a uniformed colored mesh.

Note that this script requires trimesh. 

## Existing demo and scripts

**Everything should be run after type in `sim_mode` in terminal window.**

### Running demo in simulator

Please run `sim_main/siemens_demo.py` for a demo.

Flag: -auto

If you activate -auto flag, the script would automatically spawn object at random, and clean up all objects then respawn after 3 grasp attempts.


### Forming a dataset for synthesized images

One part of this process will be to form a dataset of images (from the HSR's
camera) of the setup. The setup involves varying the amount and location of
objects that the HSR will need to pick up.

To do this:
  - if you are looking for a script that automatically randomly generates objects and take pictures, use `sim_main/dataset_collect.py`;
  - otherwise if you hope to manually adjust poses via GUI, please use `quick_img_colle/get_hsr_img.py`. It is up-to-date with current robot interface. 

#### Generate segmentation masks and bounding box labels for synthesized images

Run `sim_main/dataset_collect_segment.py` under `sim_mode`; the script would result in a dataset in `sim_img_seg/` with folder structure of:
```
sim_img_seg/
- [picture_number]/
-- rgb_[timstamp].png
-- depth_[timstamp].png
-- rgb_[max timstamp].json
-- rgb_[max timstamp].xml
```
where `rgb_[timstamp]` and `depth_[timstamp]` are the images we are investigating. The rest of the images are included for debugging purposes.

Import `segmentation` if collecting data on white background; if collecting data on a floorplan, import `segmentation_rgb`(under testing)

#### Remarks

To run datacollection properly, three variables should be properly taken care of: 

1. `spawn_object_script.py` line 11 variable `MODEL_PATH` should contain the correct absolute path pointing to the folder with the object models

2. `spawn_object_script.py` line 13 variable `MODEL_TYPE` should contain a dictionary of key:classname and value:count

3. `segmentation.py`/`segmentation_rgb.py` line 17 `SEG_LABELS` should contain a dictionary of key:classname and value:pixel value assigned to that class.



## Misc

### Gripper collision model

If we use meshes for object or gripper, we might observe strange collision dynamic in grasping. Right now we adapt a bounding primary shape (box/cylinder) collsion model on both the gripper and object models to resolve that problem.

For object models, the machnism is already encoded in the sdf parser. However, for the gripper, you might want to adjust the urdf file for hand to achieve that. A reference hand model file is included under reference for ros-kinetic and Gazebo 7.x. Please check the file to get a sense of what should be done if you are using ros-indigo.

For ros-kinetic users, the hand model file is under `/opt/ros/kinetic/share/hsrb_description/urdf/hand_v0`.