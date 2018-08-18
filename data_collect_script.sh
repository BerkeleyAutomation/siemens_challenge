#!/bin/bash


for value in {1..200}
do
	roslaunch ./sim_world/office_env.launch &
	sleep 20
	python ./sim_main/dataset_collect_segment.py 
	pkill -P $$
	sleep 30
done
