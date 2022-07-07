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
from tools.lr_scheduler import WarmUpLR
from tools.optimizer import get_learning_rate
from tools.metrics import init_metric, update_metric, get_mean_metric

from nets.util.misc import NestedTensor
from dataloader.Gold166 import detection_collate
from dataloader.Gold166 import TrainDataset, TestDataset
from nets.nrtr import build
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
    parser.add_argument('--in_checkpoint_model_path', default="None", help='path where to load checkpoint')
    parser.add_argument('--in_pretrain_model_path', default="None", help='path where to load checkpoint')
    parser.add_argument('--out_checkpoint_dir', default="None", help='path where to save checkpoint')
    parser.add_argument('--device', default='cuda',  help='device to use for training / testing')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # Test     
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--in_eval_model_path', default="None", help='model path to test')
    parser.add_argument('--out_swc_dir', default="None", help='directory to out swc')
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='DataParallel')
    return parser

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    args):
    model.train()
    criterion.train()
    metric_iters = init_metric(type="train")
    
    for _, batch in enumerate(data_loader):
        image, target = batch["image"], batch["swc_target"]
        image = image.to(args.device)
        target = [{k: v.to(args.device) for k, v in t.items()} for t in target]
        if len(target[0]["spheres"].size()) == 1:
            continue
        nest = NestedTensor(tensor=image, mask=None)
        output = model(nest)
        loss_dict = criterion(output, target)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        loss_value = losses_reduced_scaled.item()
        class_error = loss_dict_reduced['class_error'].item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric = {"loss_value": loss_value, "class_error": class_error}
        metric_iters = update_metric(metric_iters, metric)
        
        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
    metric_epoch = get_mean_metric(metric_iters)
    return metric_epoch

@torch.no_grad()
def evaluate_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,  args):
    model.eval()
    criterion.eval()
    metric_iters = init_metric(type="val")
    for _, batch in enumerate(data_loader):
        image, target = batch["image"], batch["swc_target"]
        image = image.to(args.device)
        target = [{k: v.to(args.device) for k, v in t.items()} for t in target]
        if len(target[0]["spheres"].size()) == 1:
            continue
        nest = NestedTensor(tensor=image, mask=None)
        output = model(nest)
        loss_dict = criterion(output, target)
        weight_dict = criterion.weight_dict
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        class_error = loss_dict_reduced['class_error'].item()
        
        metric = {"loss_value": loss_value, "class_error": class_error}
        metric_iters = update_metric(metric_iters, metric)
    metric_epoch = get_mean_metric(metric_iters)
    return metric_epoch

@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, data_loader: Iterable,  device, args):
    model.eval()
    metric_iters = init_metric(type="test")
    for _, batch in enumerate(data_loader):
        image_ids, image, target_mat = batch["id"], batch["image"], batch["mat_target"]
        image = image.to(device)
        output = model(NestedTensor(tensor=image, mask=None))

        prediction_MAT = calculate_predict_patches(image_ids=image_ids, output=output, args=args) 
        target_MAT = calculate_target_patches(target_mat=target_mat)
        
        metric_patch = calculate_metric_patches(predict=prediction_MAT, target=target_MAT)
        metric_iters["patch"] = update_metric(metric_iters=metric_iters["patch"], metric=metric_patch)
        metric_iters["image"] = calculate_metric_images(predict=prediction_MAT, target=target_MAT, metric_iters_image=metric_iters["image"])
        
    metric_epoch = dict()
    metric_epoch["patch"] = get_mean_metric(metric_iters["patch"])
    metric_epoch["image"] = dict()
    metric_epoch["image"]["PEC"] = (metric_iters["image"]["intersection"] / metric_iters["image"]["pred"]).cpu()
    metric_epoch["image"]["REC"] = (metric_iters["image"]["intersection"] / metric_iters["image"]["tgt"]).cpu()
    metric_epoch["image"]["F-1"] = ((2 * metric_epoch["image"]["PEC"] * metric_epoch["image"]["REC"]) / (metric_epoch["image"]["PEC"] + metric_epoch["image"]["REC"])).cpu()
    metric_epoch["image"]["JAC"] = (metric_iters["image"]["intersection"] / (metric_iters["image"]["pred"] + metric_iters["image"]["tgt"] - metric_iters["image"]["intersection"])).cpu()
    return metric_epoch  
        
@torch.no_grad()
def calculate_metric_images(predict, target, metric_iters_image):
    intersection = (predict * target)
    metric_iters_image["pred"] = metric_iters_image["pred"] + predict.sum()
    metric_iters_image["tgt"] = metric_iters_image["tgt"] + target.sum()
    metric_iters_image["intersection"] = metric_iters_image["intersection"] + intersection.sum()
    return metric_iters_image

@torch.no_grad()
def calculate_metric_patches(predict, target):
    metric = dict()
    smooth = 1
    ## RECALL
    size = predict.size(0)
    m1 = predict.view(size, -1)
    m2 = target.view(size, -1)
    intersection = (m1 * m2)
    recall_tensor = (intersection.sum(1) + smooth) / (m2.sum(1) + smooth)
    metric["REC"] = recall_tensor.sum() / target.size(0)
    ## PEC
    precision_tensor = (intersection.sum(1) + smooth) / (m1.sum(1) + smooth)
    metric["PEC"] = precision_tensor.sum() / target.size(0)
    ## F1
    metric["F-1"] = (2. * metric["PEC"] * metric["REC"]) / (metric["PEC"] + metric["REC"])
    ## Jaccard
    Jaccard_tensor = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
    metric["JAC"] = Jaccard_tensor.sum() / target.size(0)
    return metric

@torch.no_grad()
def calculate_predict_patches(image_ids, output, args):
    softmax = torch.nn.Softmax(dim=2)
    pred_class = softmax(output['pred_logits'])
    pred = torch.cat((pred_class, output['pred_boxes']), dim=2) 
    pred = pred.detach().cpu().numpy()
    pred_MAT = numpy.zeros(shape=(len(image_ids), args.crop_size, args.crop_size, args.crop_size), dtype=numpy.int)
    pred_MAT = swc2mat(image_ids, pred, pred_MAT, args.crop_size)  
    return torch.from_numpy(pred_MAT)

@torch.no_grad()
def calculate_target_patches(target_mat):
    return target_mat

def main(args):
    ## Training metric upload and Random seed
    if args.eval:
        pass
    else:
        if args.use_wandb and (args.eval == False):
            wandb.init(project="NRTR_Gold166_Neuron_Reconstruction", entity="aphrad")
        # fix the seed for reproducibility
        if args.seed != -1:
            torch.manual_seed(args.seed)
            numpy.random.seed(args.seed)
            random.seed(args.seed)

    ## Training and Test proprocess    
    device = torch.device(args.device)
    model, criterion = build(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DataParallel(model, device_ids=[0])
        model_without_ddp = model.module
    
    if args.eval:
        if args.in_eval_model_path is not "None":
            model_without_ddp.load_state_dict(torch.load(args.in_eval_model_path)["model"])
        else:
            print("Eval_model_path is None")
            sys.exit(1)
    else:
        if args.in_pretrain_model_path is not "None":
            print("Model Init")
            model.load_state_dict(torch.load(args.in_pretrain_model_path))
        else:
            print("Random Init")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
             "lr"   : args.lr_backbone}]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        WarmUp_lr_scheduler = WarmUpLR(optimizer, total_iters=args.warmup_epoch)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs - args.warmup_epoch, eta_min=0.00001, last_epoch=-1, verbose=False)

        if args.in_checkpoint_model_path is not "None":
            checkpoint = torch.load(args.in_checkpoint_model_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        print(args.epochs - args.start_epoch)
        print(optimizer)
        print(lr_scheduler)
    
    ## Train and Test Dataset
    if args.eval:
        test_dataset = TestDataset(path=args.test_dataset_path)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=detection_collate,
            drop_last=False)
    else:
        train_dataset = TrainDataset(path=args.train_dataset_path)
        test_dataset = TestDataset(path=args.test_dataset_path)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=detection_collate,
            drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=detection_collate,
            drop_last=False)

    if args.eval:
        print("Start Test")
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
        print(args.in_eval_model_path)
        metric = test_one_epoch(model=model, data_loader=test_loader, device=device, args=args)
        print(metric)
        print("Finish Test")
    else:
        print("Start Train")
        record_path = args.out_checkpoint_dir + f'/record.txt'
        record = open(record_path, "a", encoding="utf-8")
    
        for epoch in range(args.start_epoch, args.epochs):
            print("Epoch:{}/{}".format(epoch, args.epochs))
            metrics = dict()
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
            metrics["train"] = train_one_epoch(model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer, args=args)
            print("Finish Training")
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
            metrics["val"] = evaluate_one_epoch(model=model, criterion=criterion, data_loader=test_loader, args=args)
            print("Finish Validation")
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
            metrics["test"] = test_one_epoch(model=model, data_loader=test_loader, device=device, args=args)
            print("Finish Test")
            
            if epoch < args.warmup_epoch:
                WarmUp_lr_scheduler.step()
            else:
                lr_scheduler.step()
            
            record.write("Epoch:{}\n".format(epoch))
            record.write("train_LOS:{}, train_CLS={}\n".format(metrics["train"]["loss_value"], metrics["train"]["class_error"]))
            record.write("validation_LOS:{}, validation_CLS={}\n".format(metrics["val"]["loss_value"], metrics["val"]["class_error"]))
            record.write("patch_PEC:{}, patch_REC={}, patch_F-1={}, patch_JAC={}\n".format(metrics["test"]["patch"]["PEC"], 
                                                                                            metrics["test"]["patch"]["REC"],
                                                                                            metrics["test"]["patch"]["F-1"],
                                                                                            metrics["test"]["patch"]["JAC"]))
            record.write("image_PEC:{}, image_REC={}, image_F-1={}, image_JAC={}\n".format(metrics["test"]["image"]["PEC"], 
                                                                                            metrics["test"]["image"]["REC"],
                                                                                            metrics["test"]["image"]["F-1"],
                                                                                            metrics["test"]["image"]["JAC"]))
            
            if args.use_wandb:
                if (epoch - args.start_epoch) % ((args.epochs - args.start_epoch)//args.epochs) == 0:
                    lr = get_learning_rate(optimizer)
                    wandb.log({
                        "lr": lr,
                        "train_LOS" : metrics["train"]["loss_value"],
                        "train_CLS" : metrics["train"]["class_error"],
                        "valid_LOS" : metrics["val"]["loss_value"],
                        "valid_CLS" : metrics["val"]["class_error"],
                        
                        "test_patch_PEC" : metrics["test"]["patch"]["PEC"],
                        "test_patch_REC" : metrics["test"]["patch"]["REC"],
                        "test_patch_F-1" : metrics["test"]["patch"]["F-1"],
                        "test_patch_JAC" : metrics["test"]["patch"]["JAC"],
                        
                        "test_image_PEC" : metrics["test"]["image"]["PEC"],
                        "test_image_REC" : metrics["test"]["image"]["REC"],
                        "test_image_F-1" : metrics["test"]["image"]["F-1"],
                        "test_image_JAC" : metrics["test"]["image"]["JAC"]
                        }
                        )

            if args.out_checkpoint_dir is not "None":
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % 20 == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_path = args.out_checkpoint_dir + f'/checkpoint{epoch:04}.pth'
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        
        record.close()
        torch.save(model.state_dict(), args.save_model_path + '_' + date.today().strftime('%Y_%m_%d'))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)