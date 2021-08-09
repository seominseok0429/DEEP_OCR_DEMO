"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import TEXT_DETECTION.craft_utils as craft_utils
import TEXT_DETECTION.imgproc as imgproc
import TEXT_DETECTION.file_utils as file_utils
import json
import zipfile

from TEXT_DETECTION.craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class text_detection(object): 
    def __init__(self, use_cpu=False):
        self.net = CRAFT()
        if use_cpu == False:
            self.net.load_state_dict(copyStateDict(torch.load('./PRETRAINED_WEIGHT/text_detection.pth')))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
            self.net.eval()
        else:
            self.net.load_state_dict(copyStateDict(torch.load('./PRETRAINED_WEIGHT/text_detection.pth', map_location='cpu')))
            self.net.eval()

    def __call__(self, img):
        image = imgproc.loadImage(img)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        x = x.cuda()
        with torch.no_grad():
            y, feature = self.net(x)
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        return boxes, polys

if __name__ == '__main__':
    a = text_detection()
    bb, ccc = a('1.png')
    print(bb)
