# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:19:45 2021

@author: dwinant
"""


import torch
from tqdm import tqdm
import torch.nn as nn
import stiefel_optimizer
from dataloader_rkm import get_dataloader
from utils_rkm import Lin_View
import os


# Uses CNN architecture of Rob Heylen from Flanders Make for AM Use Case

class AM_Enc(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(AM_Enc, self).__init__()  # inheritance used here.
        self.args = args
        # self.main = nn.Sequential(
        #     nn.Conv2d(nChannels, self.args.capacity, **cnn_kwargs[0]),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.Conv2d(self.args.capacity, self.args.capacity * 2, **cnn_kwargs[0]),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.Conv2d(self.args.capacity * 2, self.args.capacity * 4, **cnn_kwargs[1]),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.Flatten(),
        #     nn.Linear(self.args.capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
        # )
        self.main = nn.Sequential(
            nn.Conv2d(in_channels = nChannels, out_channels = 32, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Flatten(),
            nn.Linear(768, self.args.x_fdim2),
            )
        
    def forward(self, x):
        return self.main(x)


class AM_Dec(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(AM_Dec, self).__init__()
        self.args = args
        # self.main = nn.Sequential(
        #     nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(self.args.x_fdim1, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

        #     nn.ConvTranspose2d(self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.ConvTranspose2d(self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
        #     nn.Sigmoid()
        # )
        self.main = nn.Sequential(
# =============================================================================
# 
#             nn.Linear(self.args.x_fdim2, 768),
#             Lin_View(128,2,3),
#             nn.Upsample(size = [128,5,7]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
#             nn.Upsample(size = [64,18,23]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
#             nn.Upsample(size = [64,44,54]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1),
#             nn.Upsample(size = [32,95,116]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels = 32, out_channels = nChannels, kernel_size = 3, stride = 1),
# =============================================================================
              nn.Linear(self.args.x_fdim2, 768),
              Lin_View(128,2,3),
              nn.ConvTranspose2d(in_channels = 128, out_channels = 128,kernel_size = 2, stride=2, output_padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64,kernel_size = 2, stride= 2, output_padding = (0,1)),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride=2),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1),
              nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 2, stride=2),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1),
              nn.ReLU(),
              nn.ConvTranspose2d(in_channels = 32, out_channels = nChannels, kernel_size = 3, stride = 1),

        )

    def forward(self, x):
        return self.main(x)


class AM_RKM_Stiefel(nn.Module):
    """ Defines the Stiefel RKM model and its loss functions """
    def __init__(self, ipVec_dim, args, nChannels=1, recon_loss=nn.MSELoss(reduction='sum'), ngpus=1):
        super(AM_RKM_Stiefel, self).__init__()
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels
        self.recon_loss = recon_loss

        # Initialize manifold parameter
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2)))

        # Settings for Conv layers
        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim <= 28*28*3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 8

        self.encoder = AM_Enc(self.nChannels, self.args, self.cnn_kwargs)
        self.decoder = AM_Dec(self.nChannels, self.args, self.cnn_kwargs)

    def forward(self, x):
        op1 = self.encoder(x)  # features
        op1 = op1 - torch.mean(op1, dim=0)  # feature centering
        C = torch.mm(op1.t(), op1)  # Covariance matrix

        """ Various types of losses as described in paper """
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            x_tilde2 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                    self.recon_loss(x_tilde2.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))
                    + self.recon_loss(x_tilde2.view(-1, self.ipVec_dim),
                                      x_tilde1.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'noisyU':
            x_tilde = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'deterministic':
            x_tilde = self.decoder(torch.mm(op1, torch.mm(self.manifold_param.t(), self.manifold_param)))
            f2 = self.args.c_accu * 0.5 * (self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim)))/x.size(0)  # Recons_loss

        f1 = torch.trace(C - torch.mm(torch.mm(self.manifold_param.t(), self.manifold_param), C))/x.size(0)  # KPCA
        return f1 + f2, f1, f2

# Accumulate trainable parameters in 2 groups. 1. Manifold_params 2. Network param
def param_state(model):
    param_g, param_e1 = [], []
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'manifold_param':
            param_e1.append(param)
        elif name == 'manifold_param':
            param_g.append(param)
    return param_g, param_e1

def stiefel_opti(stief_param, lrg=1e-4):
    dict_g = {'params': stief_param, 'lr': lrg, 'momentum': 0.9, 'weight_decay': 0.0005, 'stiefel': True}
    return stiefel_optimizer.AdamG([dict_g])  # CayleyAdam

def final_compute(model, args, ct, device=torch.device('cuda'), path_to_data = None):
    """ Utility to re-compute U. Since some datasets could exceed the GPU memory limits, some intermediate
    variables are saved  on HDD, and retrieved later"""
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args, path_to_data)

    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        torch.save({'oti': model.encoder(sample_batch[0].to(device))},
                   'oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))

    # Load feature-vectors
    ot = torch.Tensor([]).to(device)
    for i in range(0, len(x)):
        ot = torch.cat((ot, torch.load('oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))['oti']), dim=0)
    #os.removedirs("oti/")

    ot = (ot - torch.mean(ot, dim=0)).to(device)  # Centering
    u, _, _ = torch.svd(torch.mm(ot.t(), ot))
    u = u[:, :args.h_dim]
    with torch.no_grad():
        model.manifold_param.masked_scatter_(model.manifold_param != u.t(), u.t())
    return torch.mm(ot, u.to(device)), u
