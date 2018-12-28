#!/usr/bin/python
# -*- coding: UTF -8 -*-

import os
import random
import numpy as np
import json
import cv2

from shutil import copyfile
# For camera projection (with distortion)
from panoptic_toolbox.panutils import projectPoints

# init key numbers
CAM_NUM = 31
GET_NUM = 10

#set keypoints number
KEY_NUM = 19
# init skeleton
skeleton = [[14, 13], [13, 12], [8, 7], [7, 6], [12, 2], [6, 2], [2, 0], [0, 9],
             [0, 3], [3, 4], [4, 5], [9, 10], [10, 11], [0, 1], [1, 15], [15, 16],
             [1, 17], [17, 18]]
#init the keypoints name
keypoints0 = ['neck', 'headtop', 'bodycenter', 'Rshoulder', 'Relbow', 'Rwrist', 'Rhip', 'Rknee', 'Rankle',
              'Lshoulder', 'Lelbow', 'Lwrist', 'Lhip', 'Lknee', 'Lankle', 'Reye', 'Rear', 'Leye', 'Lear']

# init object json files contents
# add categories of dists
dic = {}
dic['images'] = []
categories = {}
dic['annotations'] = []
categories['keypoints'] = keypoints0
categories['skeleton'] = skeleton
categories['id'] = 1
dic['categories'] = []
dic['categories'].append(categories)

# set up initial path
data_path = '/home/joe/panoptic-toolbox/'
seq_name = '171204_pose1_sample'
hd_img_path = data_path + seq_name + '/hdImgs/'
hd_skel_json_path = data_path + seq_name + '/hdPose3d_stage1_coco19'
hd_skel_copy_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19_copy/'
hd_copy_img_path = data_path+seq_name+'/hdImgs_copy/'

# start getting random files and pull it into copy file
os.chdir(hd_skel_json_path)
ori_json_files = os.listdir(os.getcwd())
sample_json = random.sample(ori_json_files, GET_NUM)
sample_len = len(sample_json)
isExists = os.path.exists(hd_skel_copy_json_path)
if not isExists:
    os.makedirs(hd_skel_copy_json_path)
else:
     print("dir exists!")
for i in range(sample_len):
    ori_json = hd_skel_json_path + "/" + sample_json[i]
    copy_json = hd_skel_copy_json_path + sample_json[i]
    copyfile(ori_json, copy_json)

# select related images into copy file
os.chdir(hd_img_path)
files = os.listdir(os.getcwd())
for j in range(CAM_NUM):
    image_file_path = hd_img_path + "/" + files[j]
    os.chdir(image_file_path)
    imgs = os.listdir(os.getcwd())
    copy_img_path = data_path + seq_name + "/hdImgs_copy/"
    copy_imgs_path = copy_img_path + files[j] + "/"
    isExists0 = os.path.exists(copy_img_path)
    isExists1 = os.path.exists(copy_imgs_path)
    if not isExists0:
        os.makedirs(copy_img_path)
    if not isExists1:
        os.makedirs(copy_imgs_path)
    else:
        print("image file exists!")
    for n in range(len(imgs)):
        for r in range(GET_NUM):
            if imgs[n][-12:-4] == sample_json[r][-13:-5]:
                ori_img = image_file_path + "/" + imgs[n]
                copy_img = copy_imgs_path + imgs[n]
                copyfile(ori_img, copy_img)

os.chdir(data_path)

# Load camera calibration parameters
with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'], cam['node']) : cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k, cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3, 1))

os.chdir(hd_copy_img_path)
camera_lists = os.listdir(os.getcwd())
camera_lists_len = len(camera_lists)
for m in range(camera_lists_len):

    # get camera parameters
    cam_idx = camera_lists[m][-2:]
    cam_idx0 = int(cam_idx)
    Base_Num = cam_idx0 * 100000000
    cam = cameras[(0, cam_idx0)]
    cam_idx0_path = hd_copy_img_path + camera_lists[m] + "/"

    # get image file_name and related json files
    os.chdir(cam_idx0_path)
    img_lists = os.listdir(os.getcwd())
    img_num = len(img_lists)
    for p in range(img_num):

        images = {}
        annotations = {}
        # init annotation_file
        # set up keypoints annotation
        keypoints = [0] * 19 * 3
        # get the frame number
        hd_idx = int(img_lists[p][-12:-4])
        # write file name and id
        file_name = img_lists[p]
        file_id = hd_idx + Base_Num
        #read image infomation
        image = cv2.imread(file_name)
        sp = image.shape
        Height = sp[0]
        Width = sp[1]
        images['height'] = Height
        images['width'] = Width
        images['file_name'] = file_name
        images['id'] = file_id
        dic['images'].append(images)

        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path + '/body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

        # Cycle through all detected bodies
        for body in bframe['bodies']:

            # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
            # where c1 ... c19 are per-joint detection confidences
            skel = np.array(body['joints19']).reshape((-1, 4)).transpose()

            # Project skeleton into view (this is like cv2.projectPoints)
            pt = projectPoints(skel[0:3, :],
                               cam['K'], cam['R'], cam['t'],
                               cam['distCoef'])

            # Show only points detected with confidence
            valid = skel[3, :] > 0.1
            X = pt[0]
            Y = pt[1]
            for i0 in range(0, len(keypoints), 3):
                keypoints[i0] = X[int(i0 / 3)]
            for i1 in range(1, len(keypoints), 3):
                keypoints[i1] = Y[int(i1 / 3)]
            for i2 in range(2, len(keypoints), 3):
                keypoints[i2] = 2

        # append key value
        annotations['category_id'] = 1
        annotations['keypoints'] = keypoints
        annotations['num_keypoints'] = KEY_NUM
        annotations['image_id'] = file_id
        annotations['id'] = file_id
        dic['annotations'].append(annotations)

# create the final json file for a sequence
final_json= hd_skel_copy_json_path + seq_name + '.json'
with open(final_json, 'w', encoding='utf-8') as j_file:
    json.dump(dic, j_file, ensure_ascii=False)