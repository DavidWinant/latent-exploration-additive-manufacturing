# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:48:37 2022

@author: dwinant
"""


import torch
from tqdm import tqdm
import torch.nn as nn
#import stiefel_optimizer
#from dataloader_rkm import get_dataloader
#from utils_rkm import Lin_View
import os
import numpy as np



from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import from_numpy, split




def get_laserprofile_dataloader(args, path_to_data = r'C:\Users\dwinant\Documents\Projects\Additive Manufacturing\Data\AI_ini_cyls_input.npy'):
    """laserprofile dataloader (100, 120) images"""

    transform = transforms.Compose([transforms.ToTensor()])

    laserprofile_data = laserprofile(path_to_data, transform=transform)
    laserprofile_loader = DataLoader(laserprofile_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(laserprofile_loader))[0].size()
    #c, x, y = next(iter(tuh_loader))[0].size()
    return laserprofile_loader, c*x*y, c


class laserprofile(Dataset):
    """laserprofile dataloader class"""
    
    lat_names = ('speed','power')
    
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        
        dataset = np.fromfile(path_to_data,  dtype=np.float32)
        dataset = np.array(dataset)
        # Grey scale images 
        dataset.shape = [10000,100,120]
        labels = np.fromfile(r'C:\Users\dwinant\Documents\Projects\Additive Manufacturing\Data\AI_ini_cyls_output.npy',
                             dtype=np.float32)
        labels = np.array(labels)
        # Respectively normalized speed and normalized power
        labels.shape=[10000,2]
        
        
        self.imgs = 1-dataset[::subsample]
        self.labels = labels[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] / 255
        if self.transform:
            sample = self.transform(sample)
        return sample.float(), self.labels[idx]

def convert_to_imshow_format(image):
    # convert from CHW to HWC
    if image.shape[0] == 1:
        return image[0, :, :]
    else:
        if np.any(np.where(image < 0)):
            # first convert back to [0,1] range from [-1,1] range
            image = image / 2 + 0.5
        return image.transpose(1, 2, 0)



class Lin_View(nn.Module):
    """ Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, c, a, b):
        super(Lin_View, self).__init__()
        self.c, self.a, self.b = c, a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.c, self.a, self.b)
        except:
            #return x.view(1, self.c, self.a, self.b)
            return x.view(256, self.c, self.a, self.b)

class create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = '{}_Trained_rkm_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))
