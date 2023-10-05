'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * References: timm, simmim, and slip
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 * https://github.com/microsoft/SimMIM/
 * https://github.com/facebookresearch/SLIP
'''

import os
import os.path
import torch
import json

from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode 
from torch.utils.data import Dataset

from timm.data import create_transform

from PIL import Image
import numpy as np

import random

def pil_loader(path: str):
    ### open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
import random
import math
import numpy as np
    
class DataAugmentation:
    def __init__(self, weak_transform, strong_transform, args, train_config):
        self.transforms = [weak_transform, strong_transform]
        return 

    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)
        return images_weak, images_strong
    


def build_transform(is_train, args, train_config=None):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    # if train_config is not None:
    if is_train:
        weak_transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),       
            transforms.RandomCrop(args.input_size),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
         
        strong_transform = create_transform(
            input_size=args.input_size,
            scale=(args.train_crop_min,1),
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=mean,
            std=std,
        )          
        transform = DataAugmentation(weak_transform, strong_transform, args, train_config)

        return transform
    
    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
        return transform