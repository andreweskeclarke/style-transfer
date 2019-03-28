from __future__ import print_function
import argparse
import imageio
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
from shared_utils import load_img, save_img, merge_images
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils

CONTENT_IMAGE_PATH = './images/content.jpg'
STYLE_IMAGE_PATH = './images/style.jpg'
STYLE_IMAGE = load_img(STYLE_IMAGE_PATH).cuda()
CONTENT_IMAGE = load_img(CONTENT_IMAGE_PATH).cuda()

def run_style_transfer(content_layers, style_layers):
    if not isinstance(content_layers, (list,)):
        content_layers = [content_layers]
    if not isinstance(style_layers, (list,)):
        style_layers = [style_layers]
    id = '.'.join(str(x) for x in style_layers) + '___' + '.'.join(str(x) for x in content_layers)
    image_dir = os.path.join('./images/', id)
    def path_for(p):
        return os.path.join(image_dir, p)

    os.makedirs(image_dir, exist_ok=True)
    save_img(CONTENT_IMAGE, path_for('transfer_step_00000.png'))
    save_img(CONTENT_IMAGE, path_for('content.png'))
    save_img(STYLE_IMAGE, path_for('style.png'))
    stylized_image = Variable(CONTENT_IMAGE.data.clone(), requires_grad=True).cuda()
    loss_layers = content_layers + style_layers
    vgg = Vgg19(loss_layers).cuda()

    content_targets = list()
    for t in vgg(CONTENT_IMAGE, content_layers):
        t.detach()
        content_targets.append(t)
    content_loss_fns = [nn.MSELoss().cuda()] * len(content_targets)

    style_targets = list()
    for t in vgg(STYLE_IMAGE, style_layers):
        t.detach()
        style_targets.append(GramMatrix()(t))
    style_loss_fns = [StyleLoss().cuda()] * len(style_targets)

    targets = content_targets + style_targets
    loss_fns = content_loss_fns + style_loss_fns
    weights = [5]*len(content_targets) + [1000]*len(style_targets)

    optimizer = optim.LBFGS([stylized_image])
    print('Optimizing... id: {}'.format(id))
    for i in range(1, 100):
        def single_step():
            optimizer.zero_grad()
            outputs = vgg(stylized_image, loss_layers)
            total_loss = sum([weights[j] * loss_fns[j](o, targets[j]) for j, o in enumerate(outputs)])
            total_loss.backward()
            return total_loss
        optimizer.step(single_step)
        print('Iteration: {}, id: {}'.format(i, id))
        save_img(stylized_image, path_for('transfer_step_{}.png'.format(str(i).zfill(5))))
    save_img(stylized_image, path_for('transfer.png'))
    merge_images(
            [path_for('content.png'), path_for('style.png'), path_for('transfer.png')],
            path_for('final.png'))


if __name__ == "__main__":
    for i in range(2,17):
        run_style_transfer(Vgg19.ALL_LAYERS[i:], Vgg19.ALL_LAYERS[:i])
        run_style_transfer(Vgg19.ALL_LAYERS[i], Vgg19.ALL_LAYERS[:i])
        run_style_transfer(Vgg19.ALL_LAYERS[i:i+2], Vgg19.ALL_LAYERS[:i])

        run_style_transfer(Vgg19.ALL_LAYERS[i:], Vgg19.ALL_LAYERS)
        run_style_transfer(Vgg19.ALL_LAYERS[i], Vgg19.ALL_LAYERS)
        run_style_transfer(Vgg19.ALL_LAYERS[i:i+2], Vgg19.ALL_LAYERS)

