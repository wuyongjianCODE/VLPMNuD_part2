#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
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
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
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
import json
def preprocess_glip_result(jsonfile='LAST_PREDICT_BBOXS_old.json',visual=False):
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    svs_file_list=os.listdir('/data1/wyj/M/datasets/MoNuSACGT/stage1_train')
    svs_file_list.sort()
    for id in range(len(cocodt_dataset_ann)-1,-1,-1):
        ann=cocodt_dataset_ann[id]
        svs_image_id_of_this_ann=ann['image_id']//16
        if svs_file_list[svs_image_id_of_this_ann] in VAL_IMAGE_IDS:
            cocodt_dataset_ann.remove(ann)
    f=open("/data1/wyj/M/datasets/COCO2backup/annotations/instances_train2017.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    if visual:
        from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
        savdir = 'val_{}'.format(jsonfile).replace('.', '_')
        os.mkdir(savdir)
        vis_dataset(cocodt_dataset, savdir)
def preprocess_trainset(jsonfile='LAST_PREDICT_BBOXS.json'):
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'r')
    cocodt_dataset=json.load(f)
    for id,ann in enumerate(cocodt_dataset['annotations']):
        if ann['area']>2500:
            cocodt_dataset.remove(ann)
    f=open("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
@logger.catch
def main(exp, args, num_gpu):
    # here:to check AP of glip
    #preprocess_glip_result()
    # preprocess_trainset()
    print('first of all, we check AP of glip trainset(or the last ):::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoDt = COCO("datasets/COCO/annotations/instances_train2017.json")
    # cocoDt = COCO("/data1/wyj/M/datasets/COCO2/annotations/instances_train2017_0085.json")
    cocoGt = COCO("datasets/COCO/annotations/instances_train2017_gt.json")
    for ann in cocoDt.dataset['annotations']:
        if not 'score' in ann:
            ann['score']=0.9
            print('????no score')
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size,gen_json=True
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    import os
    # os.system('rm datasets/COCO')
    # os.system('ln -s /data1/wyj/M/datasets/COCOtrain_as_val datasets/COCO')
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
    # os.system('rm datasets/COCO')
    # os.system('ln -s /data1/wyj/M/datasets/COCO2 datasets/COCO')
    os.system('cp YOLOX_outputs/yolox_s/train.json datasets/COCO/annotations/instances_train2017.json')
