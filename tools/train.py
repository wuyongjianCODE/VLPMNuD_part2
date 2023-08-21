#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import shutil
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()
from skimage import measure,io
import numpy as np
import json
from pycocotools.coco import COCO
VAL_IMAGE_IDS = [
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",

    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-G2-A2EK-01A-02-TSB",
    "TCGA-AY-A8YK-01A-01-TS1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-RD-A8N9-01A-01-TS1",
]
def prepare_for_MONU_GT_detection():#wuyongjian: used for OUR MONUSAC
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    for image_id in range(480):
        FILENAME_LIST=os.listdir('/data1/wyj/M/datasets/MoNuSACGT/stage1_train')
        FILENAME_LIST.sort()
        THIS_FILENAME=FILENAME_LIST[image_id//16]
        THIS_CROP_NUM=image_id%16
        THE_X=THIS_CROP_NUM%4
        THE_Y = THIS_CROP_NUM //4
        masks_dir='/data1/wyj/M/datasets/MoNuSACGT/stage1_train/'+THIS_FILENAME+'/masks/'
        crop_size=250
        original_id = THIS_FILENAME+'_crop_{}.png'.format(THIS_CROP_NUM)
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        # plt.imshow(IMO)
        print("image_id:{}---filename:{}".format(image_id, original_id))
        shutil.copyfile('/data1/wyj/M/datasets/MoNuSACCROP/images/{}'.format(original_id),'/data1/wyj/M/datasets/COCO2/train2017/%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "filename":original_id})
        for instance_mask_file in os.listdir(masks_dir):
            instance_im=io.imread(masks_dir+instance_mask_file)
            BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size]
            if np.max(BORROW_PLACE)>0:
                connection_map=measure.label(BORROW_PLACE)
                connection_map_prop=measure.regionprops(connection_map)
                box=np.array(connection_map_prop[0].bbox).tolist()
                y1,x1,y2,x2=box
                # plt.gca().add_patch(plt.Rectangle(
                #     xy=(x1, y1),
                #     width=(x2 - x1),
                #     height=(y2 - y1),
                #     edgecolor=[0, 0, 1],
                #     fill=False, linewidth=1))
                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id":1,
                        "bbox": [x1,y1,x2-x1,y2-y1],
                        "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                        "area":(x2-x1)*(y2-y1),
                        "id":k,
                        "iscrowd":0,

                    })
                k+=1
        coco.dataset["annotations"] = coco_results
        # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
        coco.dataset["categories"] = [{"id": 1, "supercategory": 'nucleus', "name": 'nucleus'}]
        # plt.show()
        # pass
    with open('datasets/COCO/annotations/instances_train2017.json', "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_MONU_GT_detection_new(generate_phase):#wuyongjian: used for OUR MONUSAC
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    for image_id in range(480):
        FILENAME_LIST=os.listdir('/data1/wyj/M/datasets/MoNuSACGT/stage1_train')
        FILENAME_LIST.sort()
        THIS_FILENAME=FILENAME_LIST[image_id//16]
        THIS_CROP_NUM=image_id%16
        THE_X=THIS_CROP_NUM%4
        THE_Y = THIS_CROP_NUM //4
        masks_dir='/data1/wyj/M/datasets/MoNuSACGT/stage1_train/'+THIS_FILENAME+'/masks/'
        crop_size=250
        original_id = THIS_FILENAME+'_crop_{}.png'.format(THIS_CROP_NUM)
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        # plt.imshow(IMO)
        print("image_id:{}---filename:{}".format(image_id, original_id))
        if THIS_FILENAME in VAL_IMAGE_IDS:
            phase='val'
        else:
            phase='train'
        if phase in generate_phase:
            shutil.copyfile('/data1/wyj/M/datasets/MoNuSACCROP/images/{}'.format(original_id),'/data1/wyj/M/datasets/COCO2/%s2017/%012d.jpg'%(phase,image_id))
            coco.dataset["images"].append({"id": image_id,
                                           "height": 250, "width": 250, "filename":original_id})
            for instance_mask_file in os.listdir(masks_dir):
                instance_im=io.imread(masks_dir+instance_mask_file)
                BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size]
                if np.max(BORROW_PLACE)>0:
                    connection_map=measure.label(BORROW_PLACE)
                    connection_map_prop=measure.regionprops(connection_map)
                    box=np.array(connection_map_prop[0].bbox).tolist()
                    y1,x1,y2,x2=box
                    x1-=2
                    y1-=2
                    x2+=2
                    y2+=2
                    # plt.gca().add_patch(plt.Rectangle(
                    #     xy=(x1, y1),
                    #     width=(x2 - x1),
                    #     height=(y2 - y1),
                    #     edgecolor=[0, 0, 1],
                    #     fill=False, linewidth=1))
                    coco_results.append(
                        {
                            "image_id": image_id,
                            "category_id":1,
                            "bbox": [x1,y1,x2-x1,y2-y1],
                            "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                            "area":(x2-x1)*(y2-y1),
                            "id":k,
                            "iscrowd":0,

                        })
                    k+=1
            coco.dataset["annotations"] = coco_results
            # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
            coco.dataset["categories"] = [{"id": 1, "supercategory": 'nucleus', "name": 'nucleus'}]
            # plt.show()
            # pass
    with open('datasets/COCO/annotations/instances_{}2017.json'.format(generate_phase), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_MONU_GT_detection_new_gliptrain_aware(generate_phase):#wuyongjian: used for OUR MONUSAC#CAUTION!:this function used for glip-adapter train,with key filen_name been changed
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    for image_id in range(480):
        FILENAME_LIST=os.listdir('/data1/wyj/M/datasets/MoNuSACGT/stage1_train')
        FILENAME_LIST.sort()
        THIS_FILENAME=FILENAME_LIST[image_id//16]
        THIS_CROP_NUM=image_id%16
        THE_X=THIS_CROP_NUM%4
        THE_Y = THIS_CROP_NUM //4
        masks_dir='/data1/wyj/M/datasets/MoNuSACGT/stage1_train/'+THIS_FILENAME+'/masks/'
        crop_size=250
        original_id = THIS_FILENAME+'_crop_{}.png'.format(THIS_CROP_NUM)
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        # plt.imshow(IMO)
        print("image_id:{}---filename:{}".format(image_id, original_id))
        if THIS_FILENAME in VAL_IMAGE_IDS:
            phase='val'
        else:
            phase='train'
        if phase in generate_phase:
            # shutil.copyfile('/data1/wyj/M/datasets/MoNuSACCROP/images/{}'.format(original_id),'/data1/wyj/M/datasets/COCO2/%s2017/%012d.jpg'%(phase,image_id))
            coco.dataset["images"].append({"id": image_id,
                                           "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
            for instance_mask_file in os.listdir(masks_dir):
                instance_im=io.imread(masks_dir+instance_mask_file)
                BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size]
                if np.max(BORROW_PLACE)>0:
                    connection_map=measure.label(BORROW_PLACE)
                    connection_map_prop=measure.regionprops(connection_map)
                    box=np.array(connection_map_prop[0].bbox).tolist()
                    y1,x1,y2,x2=box
                    x1-=2
                    y1-=2
                    x2+=2
                    y2+=2
                    # plt.gca().add_patch(plt.Rectangle(
                    #     xy=(x1, y1),
                    #     width=(x2 - x1),
                    #     height=(y2 - y1),
                    #     edgecolor=[0, 0, 1],
                    #     fill=False, linewidth=1))
                    coco_results.append(
                        {
                            "image_id": image_id,
                            "category_id":1,
                            "bbox": [x1,y1,x2-x1,y2-y1],
                            "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                            "area":(x2-x1)*(y2-y1),
                            "id":k,
                            "iscrowd":0,

                        })
                    k+=1
            coco.dataset["annotations"] = coco_results
            # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
            coco.dataset["categories"] = [{"id": 1, "supercategory": 'nucleus', "name": 'nucleus'}]
            # plt.show()
            # pass
    with open('datasets/COCO/annotations/TEMP_instances_{}2017.json'.format(generate_phase), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_MONU_GT_detection_new_gliptrain_aware_SLICspecific(generate_phase):#wuyongjian: used for OUR MONUSAC#CAUTION!:this function used for glip-adapter train,with key filen_name been changed
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    for image_id in range(480):
        dirp='/home/iftwo/wyj/M/datasets/MoNuSAC/stage1_train-gt80/'#'/home/iftwo/wyj/M/datasets/MoNuSAC/stage1_train_seed0716/'
        FILENAME_LIST=os.listdir(dirp)
        FILENAME_LIST.sort()
        THIS_FILENAME=FILENAME_LIST[image_id//16]
        THIS_CROP_NUM=image_id%16
        THE_X=THIS_CROP_NUM%4
        THE_Y = THIS_CROP_NUM //4
        masks_dir=dirp+THIS_FILENAME+'/masks/'
        crop_size=250
        original_id = THIS_FILENAME+'_crop_{}.png'.format(THIS_CROP_NUM)
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        # plt.imshow(IMO)
        print("image_id:{}---filename:{}".format(image_id, original_id))
        if THIS_FILENAME in VAL_IMAGE_IDS:
            phase='val'
        else:
            phase='train'
        if phase in generate_phase:
            # shutil.copyfile('/data1/wyj/M/datasets/MoNuSACCROP/images/{}'.format(original_id),'/data1/wyj/M/datasets/COCO2/%s2017/%012d.jpg'%(phase,image_id))
            coco.dataset["images"].append({"id": image_id,
                                           "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
            for instance_mask_file in os.listdir(masks_dir):
                instance_im=io.imread(masks_dir+instance_mask_file)
                BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size]
                if np.max(BORROW_PLACE)>0:
                    connection_map=measure.label(BORROW_PLACE)
                    connection_map_prop=measure.regionprops(connection_map)
                    box=np.array(connection_map_prop[0].bbox).tolist()
                    y1,x1,y2,x2=box
                    # plt.gca().add_patch(plt.Rectangle(
                    #     xy=(x1, y1),
                    #     width=(x2 - x1),
                    #     height=(y2 - y1),
                    #     edgecolor=[0, 0, 1],
                    #     fill=False, linewidth=1))
                    coco_results.append(
                        {
                            "image_id": image_id,
                            "category_id":1,
                            "bbox": [x1,y1,x2-x1,y2-y1],
                            "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                            "area":(x2-x1)*(y2-y1),
                            "id":k,
                            "iscrowd":0,

                        })
                    k+=1
            coco.dataset["annotations"] = coco_results
            # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
            coco.dataset["categories"] = [{"id": 1, "supercategory": 'nucleus', "name": 'nucleus'}]
            # plt.show()
            # pass
    with open('datasets/COCO/annotations/TEMP_instances_{}2017_SLIC.json'.format(generate_phase), "w") as f:
        json.dump(coco.dataset, f)
    with open('datasets/COCO/annotations/instances_{}2017.json'.format(generate_phase), "w") as f:
        json.dump(coco.dataset, f)
def preprocess_glip_result(jsonfile='LAST_PREDICT_BBOXS.json',visual=False):
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("/data1/wyj/M/datasets/COCO2backup/annotations/instances_train2017.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    if visual:
        from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
        savdir = 'val_{}'.format(jsonfile).replace('.', '_')
        try:
            os.mkdir(savdir)
            vis_dataset(cocodt_dataset, savdir)
        except:
            print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def preprocess_glip_result_for_single_valset(jsonfile='LAST_PREDICT_BBOXS.json',val_name='sc',visual=False):
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("/data1/wyj/M/datasets/COCO2backup/annotations/instances_val2017.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_val2017_{}.json".format(val_name),'w')
    json.dump(cocodt_dataset,f)
    if visual:
        from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
        savdir = 'val_{}'.format(jsonfile).replace('.', '_')
        try:
            os.mkdir(savdir)
            vis_dataset(cocodt_dataset, savdir)
        except:
            print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def preprocess_trainset(jsonfile='LAST_PREDICT_BBOXS.json'):
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'r')
    cocodt_dataset=json.load(f)
    for id,ann in enumerate(cocodt_dataset['annotations']):
        if ann['area']>2500:
            cocodt_dataset.pop(id)
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)

if __name__ == "__main__":
    # prepare_for_MONU_GT_detection_new('train
    # ')
    # prepare_for_MONU_GT_detection_new('val')
    # prepare_for_MONU_GT_detection_new_gliptrain_aware('train')
    # prepare_for_MONU_GT_detection_new_gliptrain_aware_SLICspecific('train')
    # preprocess_glip_result('0.193.json',visual=True)
    # os.system('mv YOLOX_outputs/yolox_s YOLOX_outputs/yolox_s_many4n')
    # print('first of all, we check AP of glip/TS trainset:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    # from pycocotools.coco import COCO
    # from pycocotools.cocoeval import COCOeval
    # cocoDt = COCO("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json")
    # cocoGt = COCO("/data1/wyj/M/datasets/COCO2backup/annotations/instances_train2017.json")
    # print('score seted as 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # for ann in cocoDt.dataset['annotations']:
    #     ann['score']=1
    # cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
