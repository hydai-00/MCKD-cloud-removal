import os
import numpy as np
import rasterio
import csv
import torch
from torch.utils.data import Dataset
from utils.feature_detectors import get_cloud_mask, get_cloud_cloudshadow_mask
import random
import glob
import cv2


def add_speckle(img, var=0.2):
    img = np.transpose(img, (1, 2, 0))
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    sigma = np.sqrt(var)
    noise = np.random.randn(*gray.shape) * sigma
    noisy_gray = gray + gray * noise
    noisy_gray = np.clip(noisy_gray, 0, 1)
    return noisy_gray


def add_speckle_2(img, L=4):
    img = np.transpose(img, (1, 2, 0))
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)

    noise = np.random.gamma(shape=L, scale=1.0 / L, size=gray.shape).astype(np.float32)

    noisy_gray = gray * noise
    noisy_gray = np.clip(noisy_gray, 0, 1)
    return noisy_gray


class AlignedDataset(Dataset):

    def __init__(self, opts, mode, is_train=True):
        self.opts = opts
        self.data_path = opts.input_data_folder
        self.mode = mode
        if self.mode == 'train':
            self.data_path = os.path.join(self.data_path, 'TrainData/TrainData')
        elif self.mode == 'val':
            self.data_path = os.path.join(self.data_path, 'ValData/ValData')
        elif self.mode == 'test':
            self.data_path = os.path.join(self.data_path, 'TestData/TestData')

        self.filelist = glob.glob(os.path.join(os.path.join(self.data_path, 'CloudLandsat_2020'), "*"))

        self.n_images = len(self.filelist)

        self.aug = self.opts.data_augmentation
        if not is_train:
            self.aug = False
        self.crop_size = self.opts.load_size
        self.is_test = self.opts.is_test
        self.KD = self.opts.KD
        self.random_sim = self.opts.random_sim

    def __getitem__(self, index):
        num = index % 4
        index = index // 4
        if self.mode == 'train' or self.mode == 'val':
            if self.aug:
                self.rand_rot = np.random.randint(0, 4)
                self.rand_flip = np.random.randint(0, 3)
            if self.crop_size != 512:
                self.top = random.randint(0, 512 - self.crop_size)
                self.left = random.randint(0, 512 - self.crop_size)
            else:
                self.top = 0
                self.left = 0
        elif self.mode == "test":
            if num == 0:
                self.top = 0
                self.left = 0
            elif num == 1:
                self.top = 0
                self.left = 512 - self.crop_size
            elif num == 2:
                self.top = 512 - self.crop_size
                self.left = 0
            elif num == 3:
                self.top = 512 - self.crop_size
                self.left = 512 - self.crop_size

        file = self.filelist[index]

        s1_path = file.replace('CloudLandsat_2020', "Sentinel-1_2020-De")
        s2_cloudfree_path = file.replace('CloudLandsat_2020', "Landsat-8_2020")
        s2_cloudy_path = file
        mask_path = file.replace('CloudLandsat_2020', "Mask")
        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')
        mask = self.get_mask_image(mask_path).astype('float32')

        if self.aug and not self.is_test:
            if not self.rand_flip == 0:
                s1_data = np.flip(s1_data, self.rand_flip)
                s2_cloudfree_data = np.flip(s2_cloudfree_data, self.rand_flip)
                s2_cloudy_data = np.flip(s2_cloudy_data, self.rand_flip)
                mask = np.flip(mask, self.rand_flip)
            if not self.rand_rot == 0:
                s1_data = np.rot90(s1_data, self.rand_rot, (1, 2))
                s2_cloudfree_data = np.rot90(s2_cloudfree_data, self.rand_rot, (1, 2))
                s2_cloudy_data = np.rot90(s2_cloudy_data, self.rand_rot, (1, 2))
                mask = np.rot90(mask, self.rand_rot, (1, 2))

        if self.KD:
            rgb = s2_cloudfree_data[[1, 2, 3], :, :]
            sim_data = np.ndarray(s1_data.shape)
            if not self.random_sim:
                sim_data[0, :, :] = add_speckle_2(rgb)
                sim_data[1, :, :] = add_speckle_2(rgb)
            else:
                n = random.choice([0, 1])
                sim_data = s1_data.copy()
                sim_data[n, :, :] = add_speckle(rgb, 0.2)
            sim_data = torch.from_numpy(sim_data.astype('float32').copy())

        s1_data = torch.from_numpy(s1_data.copy())
        cloudy_data = torch.from_numpy(s2_cloudy_data.copy())
        source_data = torch.concat((cloudy_data, s1_data), dim=0)
        target_data = torch.from_numpy(s2_cloudfree_data.copy())
        mask = torch.from_numpy(mask.copy())
        if self.KD:
            results = {'cloudy_data': cloudy_data,
                       'target': target_data,
                       'source': source_data,
                       's1_data': s1_data,
                       'mask': mask,
                       'sim_data': sim_data,
                       'file_name': file}
        else:
            results = {'cloudy_data': cloudy_data,
                       'target': target_data,
                       'source': source_data,
                       's1_data': s1_data,
                       'mask': mask,
                       'file_name': file}

        return results

    def __len__(self):
        return self.n_images * 4

    def get_mask_image(self, path):
        if self.crop_size == 512:
            src = rasterio.open(path, 'r', driver='GTiff')
            img = src.read()
            src.close()
        else:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(self.left, self.top, self.crop_size, self.crop_size)
                img = src.read(window=window)

        img[np.isnan(img)] = np.nanmean(img)
        img = img.astype(np.float32)
        img = np.where(img != 0, 1, 0).astype(np.float32)

        return img

    def get_opt_image(self, path):
        if self.crop_size == 512:
            src = rasterio.open(path, 'r', driver='GTiff')
            img = src.read()
            src.close()
        else:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(self.left, self.top, self.crop_size, self.crop_size)
                img = src.read(window=window)

        img[np.isnan(img)] = np.nanmean(img)
        img = img.astype(np.float32)

        return img

    def get_sar_image(self, path):
        if self.crop_size == 512:
            src = rasterio.open(path, 'r', driver='GTiff')
            img = src.read()
            src.close()
        else:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(self.left, self.top, self.crop_size, self.crop_size)
                img = src.read(window=window)

        img[np.isnan(img)] = np.nanmean(img)
        return img


