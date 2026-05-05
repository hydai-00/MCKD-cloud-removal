import os
import numpy as np
import rasterio
import csv
import torch
from torch.utils.data import Dataset
from utils.feature_detectors import get_cloud_mask, get_cloud_cloudshadow_mask
import random
import cv2


def add_speckle(img, var=0.2):
    img = np.transpose(img, (1, 2, 0))
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    sigma = np.sqrt(var)
    noise = np.random.randn(*gray.shape) * sigma
    noisy_gray = gray + gray * noise
    noisy_gray = np.clip(noisy_gray, 0, 1)
    return noisy_gray


def add_speckle_2(img, L=8):

    img = np.transpose(img, (1, 2, 0))
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)

    noise = np.random.gamma(shape=L, scale=1.0/L, size=gray.shape).astype(np.float32)

    noisy_gray = gray * noise
    noisy_gray = np.clip(noisy_gray, 0, 1)
    return noisy_gray,gray


def add_speckle_noise(img, L=8):
    img = np.transpose(img, (1, 2, 0))

    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    
    noise = np.random.gamma(shape=L, scale=1.0/L, size=gray.shape).astype(np.float32)
    
    noisy_gray = noise
    
    noisy_gray = np.clip(noisy_gray, 0, 1)
    
    return noisy_gray, gray
    


class AlignedDataset(Dataset):

    def __init__(self, opts, filelist, is_train=True):
        self.opts = opts

        self.filelist = filelist
        self.n_images = len(self.filelist)

        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000

        self.aug = self.opts.data_augmentation
        if not is_train:
            self.aug = False
        self.crop_size = self.opts.load_size
        self.is_test = self.opts.is_test
        self.KD = self.opts.KD
        self.random_sim = self.opts.random_sim
        self.L = 2
   
    def set_L(self,L):
        self.L=L

    def __getitem__(self, index):
        if not self.is_test:
            if self.aug:
                self.rand_rot = np.random.randint(0, 4)
                self.rand_flip = np.random.randint(0, 3)
            if self.crop_size != 256:
                self.top = random.randint(0, 256 - self.crop_size)
                self.left = random.randint(0, 256 - self.crop_size)
            else:
                self.top = 0
                self.left = 0
        else:
            self.top = 0
            self.left = 0

        fileID = self.filelist[index]

        s1_path = os.path.join(self.opts.input_data_folder, fileID[1], fileID[2])
        s2_cloudfree_path = os.path.join(self.opts.input_data_folder,
                                         fileID[1].replace('_s1', '_s2').replace('s1_', 's2_'),
                                         fileID[2].replace('_s1_', '_s2_'))
        s2_cloudy_path = os.path.join(self.opts.input_data_folder,
                                      fileID[1].replace('_s1', '_s2_cloudy').replace('s1_', 's2_cloudy_'),
                                      fileID[2].replace('_s1_', '_s2_cloudy_'))
        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')

        if self.aug and not self.is_test:
            if not self.rand_flip == 0:
                s1_data = np.flip(s1_data, self.rand_flip)
                s2_cloudfree_data = np.flip(s2_cloudfree_data, self.rand_flip)
                s2_cloudy_data = np.flip(s2_cloudy_data, self.rand_flip)
            if not self.rand_rot == 0:
                s1_data = np.rot90(s1_data, self.rand_rot, (1, 2))
                s2_cloudfree_data = np.rot90(s2_cloudfree_data, self.rand_rot, (1, 2))
                s2_cloudy_data = np.rot90(s2_cloudy_data, self.rand_rot, (1, 2))

        cloud_coverage = get_cloud_mask(s2_cloudy_data, cloud_threshold=0.2, binarize=True)
        mask = get_cloud_cloudshadow_mask(s2_cloudy_data, 0.2)
        mask[mask != 0] = 1
        mask = np.expand_dims(mask, axis=0)

        s1_data = self.get_normalized_data(s1_data, data_type=1)
        s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)
        s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)

        if self.KD:
            rgb = s2_cloudfree_data[[1, 2, 3], :, :]
            sim_data = np.ndarray(s1_data.shape)
            gray = np.ndarray(s1_data.shape)
            if not self.random_sim:
                #L = random.randint(2,10)
                sim_data[0, :, :],gray[0, :, :] = add_speckle_2(rgb,self.L)
                sim_data[1, :, :],gray[1, :, :] = add_speckle_2(rgb,self.L)
                #sim_data = gray
                
                #sim_data[0, :, :],gray[0, :, :] = add_speckle_noise(rgb,self.L)
                #sim_data[1, :, :],gray[1, :, :] = add_speckle_noise(rgb,self.L)
            
            else:
                n = random.choice([0, 1])
                sim_data = s1_data.copy()
                sim_data[n, :, :] = add_speckle(rgb, 0.2)
            sim_data = torch.from_numpy(sim_data.astype('float32').copy())
            gray_data = torch.from_numpy(gray.astype('float32').copy())

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
                       'file_name': fileID[2],
                       'cloud_coverage': cloud_coverage,
                       'gray':gray_data}
        else:
            results = {'cloudy_data': cloudy_data,
                       'target': target_data,
                       'source': source_data,
                       's1_data': s1_data,
                       'mask': mask,
                       'file_name': fileID[2],
                       'cloud_coverage': cloud_coverage}

        return results

    def __len__(self):
        return self.n_images

    def get_opt_image(self, path):
        if self.crop_size == 256:
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
        if self.crop_size == 256:
            src = rasterio.open(path, 'r', driver='GTiff')
            img = src.read()
            src.close()
        else:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(self.left, self.top, self.crop_size, self.crop_size)
                img = src.read(window=window)

        img[np.isnan(img)] = np.nanmean(img)
        return img

    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
            data_image /= self.scale

        return data_image


'''
read data.csv
'''


def get_train_val_test_filelists(listpath):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)

    csv_file.close()

    return train_filelist, val_filelist, test_filelist

