
from operator import sub
from os import close
from re import X
from turtle import pos
from matplotlib.pyplot import annotate, xkcd
import numpy
import tifffile
import h5py
import os
import random

import torch

def UpSample(image, scale_factor, mode):
    tensor = torch.from_numpy(numpy.array([[image]]).astype(numpy.float))
    upsample_func = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    upsample_tensor = upsample_func(tensor)
    return upsample_tensor.numpy()[0,0,:,:,:]

def generateTrainDataset():
    scale_factor = 8
    path_x, path_y, path_z = 64//scale_factor, 64//scale_factor, 64//scale_factor
    root_dir = "./dataset/gold166"
    train_data_dirs = ["p_checked7_janelia_flylight_part1"] 
    trainFile = h5py.File("./H5dataset/Gold166/Gold166_train_NRTR.hdf5", "w")
    
    for train_data_dir in train_data_dirs:
        one_level_dir = os.path.join(root_dir, train_data_dir)
        for idx in range(len(os.listdir(one_level_dir))):
            two_level_dir = os.path.join(one_level_dir, os.listdir(one_level_dir)[idx])
            
            image_path = os.path.join(two_level_dir, "image.tif")
            mat_target_path = os.path.join(two_level_dir, "mat_target.tif")
            swc_target_path = os.path.join(two_level_dir, "sphere_swc_target.swc")
            print(image_path)
            
            if os.path.exists(image_path) and os.path.exists(mat_target_path) and os.path.exists(swc_target_path):
                image = tifffile.imread(image_path)
                mat_target = tifffile.imread(mat_target_path)
                swc_target = open(swc_target_path)                
                mat_target[mat_target==255] = 1
            else:
                continue    
            
            max_x, max_y, max_z = image.shape
            print(image.shape)
        
            max_query, num_annotate, num_patch = 0, 0, 0
            for index_x in range(0, max_x // path_x + 1):
                for index_y in range(0, max_y // path_y + 1):
                    for index_z in range(0, max_z // path_z + 1):
                        groupname =  train_data_dir + "+" +  os.listdir(one_level_dir)[idx] + "+" + str(index_x) + "_" + str(index_y) + "_" + str(index_z)
                        lower_x, upper_x = (int)(index_x * path_x), (int)(index_x * path_x + path_x)
                        lower_y, upper_y = (int)(index_y * path_y), (int)(index_y * path_y + path_y)
                        lower_z, upper_z = (int)(index_z * path_z), (int)(index_z * path_z + path_z)

                        if upper_x > max_x:
                            lower_x, upper_x = max_x - path_x, max_x
                        if upper_y > max_y:
                            lower_y, upper_y = max_y - path_y, max_y
                        if upper_z > max_z:
                            lower_z, upper_z = max_z - path_z, max_z

                        subimage = image[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
                        submattarget = mat_target[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
                        if len(submattarget[submattarget==1]) == 0:
                            continue
                        if scale_factor == 1:
                            upsample_subimage = subimage
                            upsample_submattarget = submattarget
                        else:
                            upsample_subimage = UpSample(subimage, scale_factor=scale_factor, mode="trilinear")
                            upsample_submattarget = UpSample(submattarget, scale_factor=scale_factor, mode="nearest")
                        
                        subswctarget = list()
                        swc_target = open(swc_target_path)
                        for neuron_message in swc_target:
                            if neuron_message[0] == '#':
                                continue
                            id, type, z, y, x, r, pa = neuron_message.split()
                            if (float(x)>= lower_x) and (float(x) < upper_x):
                                if (float(y) >= lower_y) and (float(y) < upper_y):
                                    if (float(z) >= lower_z) and (float(z) < upper_z):
                                        sphere_msg = dict(x=(float(x) - lower_x) / path_x, 
                                                          y=(float(y) - lower_y) / path_y,
                                                          z=(float(z) - lower_z) / path_z,
                                                          r=(float(r) / (path_x // 2)))
                                        annotation = dict(query_id=len(subswctarget), image_id=groupname, sphere=sphere_msg)
                                        subswctarget.append(str(annotation))
                        num_patch = num_patch + 1
                        swc_target.close()
                        if max_query < len(subswctarget):
                            max_query = len(subswctarget)
                        if len(subswctarget) > 0:
                            num_annotate = num_annotate + 1
                        
                        print(groupname)
                        g = trainFile.create_group(groupname)
                        g.create_dataset("image", data=upsample_subimage)
                        g.create_dataset("swc_target", data=subswctarget)
                        g.create_dataset("mat_target", data=upsample_submattarget)
            print("train: max_query={}".format(max_query))
            print("train: rate={}".format(num_annotate / num_patch))
    trainFile.close()
    #endregion

def generateTestDataset():
    scale_factor = 8
    path_x, path_y, path_z = 64//scale_factor, 64//scale_factor, 64//scale_factor
    root_dir = "./dataset/gold166/"
    test_data_dirs = ['p_checked6_janelia_flylight_part2']   
    testFile = h5py.File("./H5dataset/Gold166/Gold166_test_NRTR.hdf5", "w")
        
    #region TestFile
    for test_data_dir in test_data_dirs:
        one_level_dir = os.path.join(root_dir, test_data_dir)
        for idx in range(len(os.listdir(one_level_dir))):
            two_level_dir = os.path.join(one_level_dir, os.listdir(one_level_dir)[idx])
            
            image_path = os.path.join(two_level_dir, "image.tif")
            mat_target_path = os.path.join(two_level_dir, "mat_target.tif")
            swc_target_path = os.path.join(two_level_dir, "sphere_swc_target.swc")
            
            if os.path.exists(image_path) and os.path.exists(mat_target_path) and os.path.exists(swc_target_path):
                image = tifffile.imread(image_path)
                mat_target = tifffile.imread(mat_target_path)
                swc_target = open(swc_target_path)
                mat_target[mat_target==255] = 1
            else:
                continue
            
            max_x, max_y, max_z = image.shape
            print(image.shape)
        
            max_query, num_annotate, num_patch = 0, 0, 0
            for index_x in range(0, max_x // path_x + 1):
                for index_y in range(0, max_y // path_y + 1):
                    for index_z in range(0, max_z // path_z + 1):
                        groupname =  test_data_dir + "+" +  os.listdir(one_level_dir)[idx] + "+" + str(index_x) + "_" + str(index_y) + "_" + str(index_z)
                        lower_x, upper_x = (int)(index_x * path_x), (int)(index_x * path_x + path_x)
                        lower_y, upper_y = (int)(index_y * path_y), (int)(index_y * path_y + path_y)
                        lower_z, upper_z = (int)(index_z * path_z), (int)(index_z * path_z + path_z)

                        if upper_x > max_x:
                            lower_x, upper_x = max_x - path_x, max_x
                        if upper_y > max_y:
                            lower_y, upper_y = max_y - path_y, max_y
                        if upper_z > max_z:
                            lower_z, upper_z = max_z - path_z, max_z

                        subimage = image[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
                        submattarget = mat_target[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z]
                        if len(submattarget[submattarget==1]) == 0:
                            continue
                        if scale_factor == 1:
                            upsample_subimage = subimage
                            upsample_submattarget = submattarget
                        else:
                            upsample_subimage = UpSample(subimage, scale_factor=scale_factor, mode="trilinear")
                            upsample_submattarget = UpSample(submattarget, scale_factor=scale_factor, mode="nearest")

                        subswctarget = list()
                        swc_target = open(swc_target_path)
                        for neuron_message in swc_target:
                            if neuron_message[0] == '#':
                                continue
                            id, type, z, y, x, r, pa = neuron_message.split()
                            if (float(x)>= lower_x) and (float(x) < upper_x):
                                if (float(y) >= lower_y) and (float(y) < upper_y):
                                    if (float(z) >= lower_z) and (float(z) < upper_z):
                                        sphere_msg = dict(x = (float(x) - lower_x) / path_x, 
                                                          y = (float(y) - lower_y) / path_y,
                                                          z = (float(z) - lower_z) / path_z,
                                                          r = (float(r) / (path_x // 2)))
                                        annotation = dict(query_id=len(subswctarget), image_id=groupname, sphere=sphere_msg)
                                        subswctarget.append(str(annotation))
                        num_patch = num_patch + 1
                        swc_target.close()
                        if max_query < len(subswctarget):
                            max_query = len(subswctarget)
                        if len(subswctarget) > 0:
                            num_annotate = num_annotate + 1

                        print(groupname)
                        g = testFile.create_group(groupname)
                        g.create_dataset("image", data=upsample_subimage)
                        g.create_dataset("swc_target", data=subswctarget)
                        g.create_dataset("mat_target", data=upsample_submattarget)
            print("test: max_query={}".format(max_query))
            print("test: rate={}".format(num_annotate / num_patch))
    testFile.close()
    #endregion

if __name__=="__main__":
    generateTrainDataset()
    generateTestDataset()