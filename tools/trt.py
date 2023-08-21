#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from skimage import io, color
import numpy as np

class Cluster(object):

    cluster_index = 1

    def __init__(self, row, col, l=0, a=0, b=0):
        self.update(row, col, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, row, col, l, a, b):
        self.row = row
        self.col = col
        self.l = l
        self.a = a
        self.b = b


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, row, col):
        row=int(row)
        col=int(col)
        return Cluster(row, col,
                       self.data[row][col][0],
                       self.data[row][col][1],
                       self.data[row][col][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.N = self.rows * self.cols
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.rows, self.cols), np.inf)

    def init_clusters(self):
        row = self.S / 2
        col = self.S / 2
        while row < self.rows:
            while col < self.cols:
                self.clusters.append(self.make_cluster(row, col))
                col+= self.S
            col = self.S / 2
            row += self.S

    def get_gradient(self, row, col):
        if col + 1 >= self.cols:
            col = self.cols - 2
        if row + 1 >= self.rows:
            row = self.rows - 2

        gradient = (self.data[row + 1][col][0] +self.data[row][col+1][0]-2*self.data[row][col][0])+ \
                   (self.data[row + 1][col][1] +self.data[row][col+1][1]-2*self.data[row][col][1]) + \
                   (self.data[row + 1][col][2] +self.data[row][col+1][2]-2*self.data[row][col][2])

        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.row, cluster.col)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _row = cluster.row + dh
                    _col = cluster.col + dw
                    new_gradient = self.get_gradient(_row, _col)
                    if new_gradient < cluster_gradient:
                        cluster.update(_row, _col, self.data[_row][_col][0], self.data[_row][_col][1], self.data[_row][_col][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.row - 2 * self.S, cluster.row + 2 * self.S):
                if h < 0 or h >= self.rows: continue
                for w in range(cluster.col - 2 * self.S, cluster.col + 2 * self.S):
                    if w < 0 or w >= self.cols: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.row, 2) +
                        math.pow(w - cluster.col, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h =int( sum_h / number)
                _w =int( sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.row][cluster.col][0] = 0
            image_arr[cluster.row][cluster.col][1] = 0
            image_arr[cluster.row][cluster.col][2] = 0
        self.save_lab_image(name, image_arr)

    def iterates(self):
        self.init_clusters()
        self.move_clusters()
        #考虑到效率和效果，折中选择迭代10次
        for i in range(10):
            self.assignment()
            self.update_cluster()
        self.save_current_image("output.jpg")


if __name__ == '__main__':
    p = SLICProcessor('beauty.jpg', 200, 40)
    p.iterates()
import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
