## tifffile read image.tif as numpy : [min_x:max_x, min_y:max_y, min_z:max_z]
## swc neuron :id, type, z, y, x, r, parent
## x \in [min_x, max_x], y \in [min_y, max_y], z \in [min_z, max_z] 

import os
import torch
import numpy as np

def preprocess():
    root_dir = "./dataset/gold166"
    
    for idx in range(len(os.listdir(root_dir))):
        one_level_dir = os.path.join(root_dir, os.listdir(root_dir)[idx]) 
        for idx in range(len(os.listdir(one_level_dir))):
            two_level_dir = os.path.join(one_level_dir, os.listdir(one_level_dir)[idx]) 
            if os.path.exists(os.path.join(two_level_dir, "swc_target.swc")):
                cylinder_swc_2_sphere_swc(two_level_dir)

def cylinder_swc_2_sphere_swc(swc_dir):
    in_swc_target = open(os.path.join(swc_dir, "swc_target.swc"), "r")
    out_swc_target = open(os.path.join(swc_dir, "sphere_swc_target.swc"), "w")
    neuron_id_table = swc_2_table(in_swc_target)
    out_index = 1
    for neuron in neuron_id_table:
        if neuron["pa"] == -1:
            neuron_msg = "{:d} {} {:.2f} {:.2f} {:.2f} {:.2f} {:d}\n".format(
                    out_index, neuron["type"], neuron["z"], neuron["y"], neuron["x"], neuron["r"], -1)
            out_index = out_index + 1
            out_swc_target.write(neuron_msg)
            continue
        else:
            next_neuron = neuron_id_table[neuron["pa"]-1]
            in_mat = np.array([[neuron["x"], next_neuron["x"]],
                               [neuron["y"], next_neuron["y"]],
                               [neuron["z"], next_neuron["z"]],
                               [neuron["r"], next_neuron["r"]]])
            out_mat = UpSample(image=in_mat, size=(4,3), mode="bilinear")
            
            for index in range(0, np.shape(out_mat)[1]-1): 
                neuron_x = out_mat[0, index]
                neuron_y = out_mat[1, index]
                neuron_z = out_mat[2, index]
                neuron_r = out_mat[3, index]
                neuron_msg = "{:d} {} {:.2f} {:.2f} {:.2f} {:.2f} {:d}\n".format(
                    out_index, neuron["type"], neuron_z, neuron_y, neuron_x, neuron_r, -1)
                out_index = out_index + 1
                out_swc_target.write(neuron_msg)
    in_swc_target.close()
    out_swc_target.close()
        
            
def UpSample(image, size, mode):
    tensor = torch.from_numpy(np.array([[image]]).astype(np.float))
    upsample_func = torch.nn.Upsample(size=size, mode=mode)
    upsample_tensor = upsample_func(tensor)
    return upsample_tensor.numpy()[0,0,:,:]

def swc_2_table(swc_target):
    neuron_id_table = list()
    for neuron_message in swc_target:
        if neuron_message[0] == '#':
            continue
        id, type, z, y, x, r, pa = neuron_message.split()
        neuron = {"id":int(id), "type":type, "z":float(z), "y":float(y), "x":float(x), "r":float(r), "pa":int(pa)}
        neuron_id_table.append(neuron)
    return neuron_id_table        

if __name__=="__main__":
    preprocess()