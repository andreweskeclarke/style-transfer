from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
from PIL import Image
from shared_utils import load_img, save_img


content_image_path = './images/content.jpg'
style_image_path = './images/style.jpg'

style_image = load_img(style_image_path).cuda()
content_image = load_img(content_image_path).cuda()
stylized_image = Variable(content_image.data.clone(), requires_grad=True).cuda()

vgg = models.vgg19(pretrained=True)
vgg = nn.Sequential(*list(vgg.children())[:-2])
for param in vgg.parameters():
    param.requires_grad = False

vgg.cuda()

class GramMatrix(nn.Module):
    def forward(self, input):
        n_batches, n_channels, height, width = input.size()
        flattened = input.view(n_batches, n_channels, height * width)
        return torch.bmm(flattened, flattened.transpose(1,2)).div_(height * width)

class StyleLoss(nn.Module):
    def forward(self, input, target):
        return nn.MSELoss()(GramMatrix()(input), target)

content_layers = [21] # Only valid for VGG19
content_targets = list([vgg[:l+1](content_image).detach() for l in content_layers])
content_loss_fns = [nn.MSELoss().cuda()] * len(content_targets)
style_layers = [1, 6, 11, 20, 29] # Only valid for VGG19
style_targets = list([GramMatrix()(vgg[:l+1](style_image)).detach() for l in style_layers])
style_loss_fns = [StyleLoss().cuda()] * len(style_targets)

loss_layers = content_layers + style_layers
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
        outputs = list([vgg[:l+1](stylized_image) for l in loss_layers]) # TODO: this repeates needless computations
        total_losses = []
        for j in range(len(outputs)):
           total_losses.append(weights[j] * loss_fns[j](outputs[j], targets[j]))
        total_loss = sum(total_losses)
        total_loss.backward()
        return total_loss
    optimizer.step(single_step)
    save_img(stylized_image.data[0].cpu().squeeze(), 'steps/transfer_{}.png'.format(str(i).zfill(5)))

final_stylized_image = stylized_image.data[0].cpu()
save_img(final_stylized_image.squeeze())
