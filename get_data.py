#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import shutil

#init
new_path = [0]*63
path_num = 0
init_path = "/data/openpose_dataset/panoptic-toolbox/"
delet = ['hdPose3d_stage1_coco19.tar','hdVideos']
f = open("/home/joe/PycharmProjects/get_openpose_data/openpose_datasets.txt", "r")
os.chdir(init_path)
for value in f.readlines():
    value_new = value.strip("\n")
    os.system("./scripts/getData.sh {}".format(value_new))
    os.system("./scripts/extractAll.sh {}".format(value_new))
    new_path[path_num] = os.path.join(init_path, value_new)
    os.chdir(new_path[path_num])
    files = os.listdir(os.getcwd())
    for file in files:
         if str(file) == delet[0]:
             os.remove(file)
         if str(file) == delet[1]:
             shutil.rmtree(file)
    path_num +=1
    os.chdir(init_path)

f.close()