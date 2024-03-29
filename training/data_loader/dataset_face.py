import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import degradations


class GFPGAN_degradation(object):
    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20/255.
    
    def degrade_process(self, img_gt):
        hflip = False
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)
            hflip = True

        h, w = img_gt.shape[:2]
       
        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq, hflip

class GPEN_degradation(GFPGAN_degradation):
    def __init__(self):
        super().__init__()
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [1, 12]
        self.noise_range = [0, 25]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20/255.

class FaceDataset(Dataset):
    def __init__(self, path, comp_path, resolution=512):
        self.resolution = resolution

        self.HQ_imgs = []
        for path, subdirs, files in os.walk(path):
            for name in files:
                self.HQ_imgs.append(os.path.join(path, name))
        if self.HQ_imgs[0].split(".")[-1] == "txt":
            self.HQ_imgs = self.HQ_imgs[1:]
        self.length = len(self.HQ_imgs)

        self.degrader = GFPGAN_degradation()
        #self.degrader = GPEN_degradation()

        self.components_list = torch.load(comp_path)
        self.eye_enlarge_ratio = 1.4

    def __len__(self):
        return self.length

    def get_component_coordinates(self, index, status):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        components_bbox = self.components_list[f'{index:08d}']
        if status:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.resolution - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.resolution - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.resolution - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        while True:
            try:
                img_gt = cv2.imread(self.HQ_imgs[index], cv2.IMREAD_COLOR)
                img_gt = cv2.resize(img_gt, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(str(e))
                index += 1
                index %= self.length
                continue
            break
        
        # BFR degradation
        # We adopt the degradation of GFPGAN for simplicity, which however differs from our implementation in the paper.
        # Data degradation plays a key role in BFR. Please replace it with your own methods.
        img_gt = img_gt.astype(np.float32)/255.
        img_gt, img_lq, hflip = self.degrader.degrade_process(img_gt)

        ind = int(self.HQ_imgs[index].split("/")[-1].split(".")[0])
        locations = self.get_component_coordinates(ind, hflip)
        loc_left_eye, loc_right_eye, loc_mouth = locations
        #locs = {"loc_left_eye": loc_left_eye, "loc_right_eye": loc_right_eye, "loc_mouth": loc_mouth}

        img_gt =  (torch.from_numpy(img_gt) - 0.5) / 0.5
        img_lq =  (torch.from_numpy(img_lq) - 0.5) / 0.5
        
        img_gt = img_gt.permute(2, 0, 1).flip(0) # BGR->RGB
        img_lq = img_lq.permute(2, 0, 1).flip(0) # BGR->RGB

        return img_lq, img_gt, loc_left_eye, loc_right_eye, loc_mouth