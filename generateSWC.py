import os
import sys
import math
import time
import torch
import wandb
import numpy
import random
import argparse
import nets.util.misc as utils

from torch._C import DisableTorchFunction
from datetime import date
from typing import Iterable
from lr_scheduler import WarmUpLR
from optimizer import get_learning_rate
from metrics import init_metric, update_metric, get_mean_metric

from nets.util.misc import NestedTensor
from dataloader.Gold166 import detection_collate
from dataloader.Gold166 import TrainDataset, TestDataset
from nets.detr import build
from utils.swc2mat import swc2mat

def get_args_parser():
    parser = argparse.ArgumentParser('NRTR_Neuron_Reconstruction_Transformer.pytorch', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--val_batch_size', default=30, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.5, type=float,
                        help='gradient clipping max norm')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                        help="If use wandb update metrics")
    
    # Model parameters
    parser.add_argument('--pretrain_model_path', type=str, default="None",
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--save_model_path', type=str, default="./checkpoint/model_last.pth",
                        help="Path to the saved model. If set, only the mask head will be trained")
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, choices=('resnet18', 'resnet34', 'resnet50'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=192, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Warmup_epoch
    parser.add_argument('--warmup_epoch', default=10, type=int,
                        help="Train segmentation head if the flag is provided")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=0.05, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=1, type=float)
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--ce_loss_coef', default=0.25, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset
    parser.add_argument('--train_dataset_path', default="./H5dataset/gold166/Gold166_train_DETR_64_0612.hdf5", type=str)
    parser.add_argument('--test_dataset_path', default="./H5dataset/gold166/Gold166_test_DETR_64_0612.hdf5", type=str)
    parser.add_argument('--crop_size', default=64, type=int)
    
    # Others output_path
    parser.add_argument('--checkpoint_path', default="None", help='path where to load checkpoint')
    parser.add_argument('--checkpoint_dir', default="None", help='path where to save checkpoint')
    parser.add_argument('--device', default='cuda',  help='device to use for training / testing')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # Test     
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_model_path', default="None", help='model path to test')
    parser.add_argument('--out_swc_dir', default="None", help='directory to out swc')
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='DataParallel')
    return parser

@torch.no_grad()
def generateSWC(model, data_loader, device, args):
    model.eval()
    swcIndexDict = dict()
    for _, batch in enumerate(data_loader):
        image_ids, image, target_mat = batch["id"], batch["image"], batch["mat_target"]
        image = image.to(device)
        output = model(NestedTensor(tensor=image, mask=None))
        if args.out_swc_dir is not "None":
            softmax = torch.nn.Softmax(dim=2)
            pred_class = softmax(output['pred_logits'])
            pred = torch.cat((pred_class, output['pred_boxes']), dim=2) 
            pred_numpy = pred.detach().cpu().numpy()

            for idx in range(len(image_ids)):
                image_id = image_ids[idx]
                # one_level_dir, patch = image_id.split("+")
                one_level_dir, two_level_dir, patch = image_id.split("+")
                    
                if os.path.exists(os.path.join(args.out_swc_dir, one_level_dir)) == False:
                    os.mkdir(os.path.join(args.out_swc_dir, one_level_dir))
                if os.path.exists(os.path.join(os.path.join(args.out_swc_dir, one_level_dir), two_level_dir)) == False:
                    os.mkdir(os.path.join(os.path.join(args.out_swc_dir, one_level_dir), two_level_dir))
                
                # out_swc_path = os.path.join(os.path.join(args.out_swc_dir, one_level_dir), "swc_target.swc")    
                out_swc_path = os.path.join(os.path.join(os.path.join(args.out_swc_dir, one_level_dir), two_level_dir), "swc_target.swc")
                swcFile = open(out_swc_path, "a")
                if out_swc_path not in swcIndexDict.keys():
                    swcIndexDict[out_swc_path] = 0
                
                patch_x, patch_y, patch_z = patch.split("_")
                patch_x, patch_y, patch_z = int(patch_x), int(patch_y), int(patch_z)
                offset_x, offset_y, offset_z = patch_x * 8, patch_y * 8, patch_z * 8
                        
                for index in range(args.num_queries):
                    if pred[idx, index, 0] >= 0.5:
                        sphere_msg = "{:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:d}\n".format(swcIndexDict[out_swc_path], 2,
                                                                                            offset_z + pred_numpy[idx, index, 4] * 8,
                                                                                            offset_y + pred_numpy[idx, index, 3] * 8,
                                                                                            offset_x + pred_numpy[idx, index, 2] * 8,
                                                                                            pred_numpy[idx, index, 5] * 8, -1)
                        swcIndexDict[out_swc_path] = swcIndexDict[out_swc_path] + 1
                        swcFile.write(sphere_msg)
                swcFile.close()
    return 

def main(args):
    ## Training and Test proprocess    
    device = torch.device(args.device)
    model, _ = build(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DataParallel(model, device_ids=[0])
        model_without_ddp = model.module
        
    if args.eval_model_path is not "None":
        model_without_ddp.load_state_dict(torch.load(args.eval_model_path)["model"])
    else:
        print("Eval_model_path is None")
        sys.exit(1)
    test_dataset = TestDataset(path=args.test_dataset_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        drop_last=False)
    print("Start Test")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
    generateSWC(model=model, data_loader=test_loader, device=device, args=args)
    print("Finish Test")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
        
if __name__=="__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)