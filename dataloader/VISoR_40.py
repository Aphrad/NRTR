# rewrited
import ast
import torch
import h5py
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
import random

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    image = []
    swc_target = []
    mat_target = []
    for sample in batch:
        image.append(sample["image"])
        swc_target.append(sample["swc_target"])
    return dict(image=torch.stack(image, 0), swc_target=swc_target, mat_target=torch.stack(mat_target, 0))

class TrainDataset(torch.utils.data.Dataset):
    """
    Returns:
        image: numpy size=[channel=1, depth=96, width=96, length=96]
        target: List[annotation] size=[num_queries]
                annotation:{
                    "query_id": int,
                    "image_id": str
                    "sphere": [x,y,z,radius]
                }
    """
    def __init__(self, path='../../dataset/VISoR_40_train_DETR_96.hdf5'):
        self.dataset_path = path
        self.dataset = h5py.File(self.dataset_path, "r")
        self.dataset_keys = list(self.dataset.keys())
        self.dataset_length = len(self.dataset_keys)
        print('%s datasets num = %s' % (path, self.dataset_length))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        data_id = self.dataset_keys[index]
        data = self.dataset.get(data_id)
            
        image, swc_target, mat_target = data["image"][()], data["swc_target"][()], data["mat_target"][()]
        swc_target = convert_target(swc_target)
        image = np.array([image])

        return dict(image=torch.tensor(image.astype(np.float32)), swc_target=swc_target, mat_target=torch.tensor(mat_target.astype(np.int)))

class ValidationDataset(torch.utils.data.Dataset):
    """
    Returns:
        image: numpy size=[channel=1, depth=96, width=96, length=96]
        target: List[annotation] size=[num_queries]
                annotation:{
                    "query_id": int,
                    "image_id": str
                    "sphere": [x,y,z,radius]
                }
    """
    def __init__(self, path='../../dataset/VISoR_40_test_DETR_96.hdf5'):
        self.dataset_path = path
        self.dataset = h5py.File(self.dataset_path, "r")
        self.dataset_keys = list(self.dataset.keys())
        self.dataset_length = len(self.dataset_keys)
        print('%s datasets num = %s' % (path, self.dataset_length))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        data_id = self.dataset_keys[index]
        data = self.dataset.get(data_id)
            
        image, swc_target, mat_target = data["image"][()], data["swc_target"][()], data["mat_target"][()]
        swc_target = convert_target(swc_target)
        image = np.array([image])

        return dict(image=torch.tensor(image.astype(np.float32)), swc_target=swc_target, mat_target=torch.tensor(mat_target.astype(np.int)))

class TestDataset(torch.utils.data.Dataset):
    """
    Returns:
        image: numpy size=[channel=1, depth=96, width=96, length=96]
        target: List[annotation] size=[num_queries]
                annotation:{
                    "query_id": int,
                    "image_id": str
                    "sphere": [x,y,z,radius]
                }
    """
    def __init__(self, path='../../dataset/VISoR_40_test_DETR.hdf5'):
        self.dataset_path = path
        self.dataset = h5py.File(self.dataset_path, "r")
        self.dataset_keys = list(self.dataset.keys())
        self.dataset_length = len(self.dataset_keys)
        print('%s datasets num = %s' % (path, self.dataset_length))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        data_id = self.dataset_keys[index]
        data = self.dataset.get(data_id)
            
        image_id, image, swc_target, mat_target = data_id, data["image"][()], data["swc_target"][()], data["mat_target"][()]
        image = np.array([image])
        mat_target = np.array([mat_target])

        return dict(id=image_id, image=torch.tensor(image.astype(np.float32)), mat_target=mat_target)

def convert_target(target_str):
    target = dict()
    if len(target_str) == 0:
        target["classes"] = torch.tensor([])
        target["spheres"] = torch.tensor([])
    else:
        for annotation in target_str:
            annotation = eval(annotation)
            if "spheres" in target:
                sphere = torch.tensor([[float(annotation["sphere"]["x"]),
                                        float(annotation["sphere"]["y"]),
                                        float(annotation["sphere"]["z"]),
                                        float(annotation["sphere"]["r"])]])
                target["spheres"] = torch.cat((target["spheres"], sphere))
                target["classes"] = torch.cat((target["classes"], torch.tensor(np.array([0]).astype(np.long))))
            else:
                target["spheres"] = torch.tensor([[float(annotation["sphere"]["x"]),
                                                float(annotation["sphere"]["y"]),
                                                float(annotation["sphere"]["z"]),
                                                float(annotation["sphere"]["r"])]])
                target["classes"] = torch.tensor(np.array([0]).astype(np.long))
    return target

if __name__ == "__main__":
    dataset = TrainDataset()
    for i in range(len(dataset)):
        image, label = dataset.__getitem__(i)
