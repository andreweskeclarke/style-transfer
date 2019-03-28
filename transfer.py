from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from losses import GramMatrix, StyleLoss
from models import Vgg19
from shared_utils import load_img, save_img
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils

def main():
    content_image_path = './images/content.jpg'
    style_image_path = './images/style.jpg'
    style_image = load_img(style_image_path).cuda()
    content_image = load_img(content_image_path).cuda()
    stylized_image = Variable(content_image.data.clone(), requires_grad=True).cuda()
    content_layers = [22] # Only valid for VGG19
    style_layers = [1, 6, 11, 20, 29] # Only valid for VGG19
    loss_layers = content_layers + style_layers
    vgg = Vgg19(loss_layers)
    for param in vgg.parameters():
        param.requires_grad = False

    content_targets = list(vgg(content_image, content_layers))
    style_targets = list([GramMatrix()(t) for t in vgg(style_image, style_layers)])

    content_loss_fns = [nn.MSELoss().cuda()] * len(content_targets)
    style_loss_fns = [StyleLoss().cuda()] * len(style_targets)

    targets = content_targets + style_targets
    loss_fns = content_loss_fns + style_loss_fns
    weights = [5]*len(content_targets) + [1000]*len(style_targets)
    optimizer = optim.LBFGS([stylized_image])
    n_iterations = 100
    for i in range(1, n_iterations):
        save_img(content_image.data[0].cpu().squeeze(), 'steps/transfer_00000.png')
        print('Iteration: {}'.format(i))
        def single_step():
            optimizer.zero_grad()
            outputs = vgg(stylized_image, loss_layers)
            total_losses = []
            for j in range(len(outputs)):
               total_losses.append(weights[j] * loss_fns[j](outputs[j], targets[j]))
            total_loss = sum(total_losses)
            total_loss.backward()
            return total_loss
        optimizer.step(single_step)
        save_img(stylized_image.data[0].cpu().squeeze(), 'steps/transfer_{}.png'.format(str(i).zfill(5)))

if __name__ == "__main__":
    main()
