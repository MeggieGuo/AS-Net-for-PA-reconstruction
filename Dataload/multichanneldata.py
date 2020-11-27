import numpy as np
import torch
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as scio
from PIL import Image


def np_range_norm(image, maxminnormal=True, range1=True):
    # if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):

    if maxminnormal:
        _min = image.min()
        _range = image.max() - image.min()
        narmal_image = (image - _min) / _range
        if range1:
            narmal_image = (narmal_image - 0.5) * 2
    else:
        _mean = image.mean()
        _std = image.std()
        narmal_image = (image - _mean) / _std

    return narmal_image

class ReconDataset0526(data.Dataset):
    __inputdata = []
    __outputdata = []
    __inputimg = []

    def __init__(self, root, train=True, transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__inputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"

        for file in os.listdir(folder):
            matdata = scio.loadmat(folder + file)
            self.__inputdata.append(np.transpose(matdata['sparse_sensor_data'])[np.newaxis, :, :])
            self.__outputdata.append(matdata['p0'][np.newaxis, :, :])
            self.__inputimg.append(matdata['p0_das'][np.newaxis, :, :])

    def __getitem__(self, index):

        rawdata = self.__inputdata[index]
        reconstruction = self.__outputdata[index]
        beamform = self.__inputimg[index]
        rawdata = rawdata[:, :1792, :].reshape((128,14,128)).swapaxes(0,1)
        rawdata = torch.Tensor(rawdata)
        reconstructions = torch.Tensor(reconstruction)
        beamform = torch.Tensor(beamform)

        return rawdata, reconstructions, beamform

    def __len__(self):
        return len(self.__inputdata)

class FishData(data.Dataset):
    __inputdata = []
    __outputdata = []
    __inputimg = []

    def __init__(self, root, train=True, transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__inputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"

        for file in os.listdir(folder):
            matdata = scio.loadmat(folder + file)
            self.__inputdata.append(np.transpose(matdata['sparse_sensor_data'])[np.newaxis, :, :])
            self.__outputdata.append(matdata['pgt'][np.newaxis, :, :])
            self.__inputimg.append(matdata['pdas'][np.newaxis, :, :])

    def __getitem__(self, index):

        rawdata = self.__inputdata[index]
        reconstruction = self.__outputdata[index]
        reconstruction = np_range_norm(reconstruction,maxminnormal=True, range1=False)
        beamform = self.__inputimg[index]
        beamform = np_range_norm(beamform, maxminnormal=True, range1=False)

        c = np.zeros((1, 36, 128))
        rawdata = np.concatenate([rawdata, c], axis=1).reshape((128,12,128)).swapaxes(0,1)

        rawdata = torch.Tensor(rawdata)
        reconstructions = torch.Tensor(reconstruction)
        beamform = torch.Tensor(beamform)

        return rawdata, reconstructions, beamform

    def __len__(self):
        return len(self.__inputdata)

class HisMiceData(data.Dataset):
    __inputdata = []
    __outputdata = []
    __inputimg = []

    def __init__(self, root, train=True, transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__inputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"

        for file in os.listdir(folder):
            matdata = scio.loadmat(folder + file)
            self.__inputdata.append(np.transpose(matdata['sparse_sensor_data'])[np.newaxis, :, :])
            self.__outputdata.append(matdata['pgt'][np.newaxis, :, :])
            self.__inputimg.append(matdata['pdas'][np.newaxis, :, :])

    def __getitem__(self, index):

        rawdata = self.__inputdata[index]
        reconstruction = self.__outputdata[index]  # .reshape((1,1,2560,120))
        reconstruction = np_range_norm(reconstruction, maxminnormal=True, range1=False)
        beamform = self.__inputimg[index]
        beamform = np_range_norm(beamform, maxminnormal=True, range1=False)

        c = np.zeros((1, 36, 128))
        rawdata = np.concatenate([rawdata, c], axis=1).reshape((128, 12, 128)).swapaxes(0, 1)  # (20,128,128)
        
        rawdata = torch.Tensor(rawdata)
        reconstructions = torch.Tensor(reconstruction)
        beamform = torch.Tensor(beamform)

        return rawdata, reconstructions, beamform

    def __len__(self):
        return len(self.__inputdata)
