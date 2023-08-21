
import os
for epochn in range(10,310,10):
    fname='epoch_{}_ckpt.pth'.format(epochn)
    os.system('/home/iftwo/anaconda3/envs/torch/bin/python tools/eval.py -n yolox-s -c /data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_from008to246/{} -b 1 -d 1 --conf 0.001 --fp16 --fuse'.format(fname))