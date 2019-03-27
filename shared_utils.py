# Copyright (c) 2018 Pawan Sasanka Ammanamanchi and licensed under the MIT LIcense
# Available at https://github.com/Shashi456/Neural-Style

import torch
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
from PIL import Image

#Dataset Processing
transform_img = transforms.Compose([
    transforms.Resize(512), #Default image_size
    transforms.ToTensor(), #Transform it to a torch tensor
    transforms.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]), #subracting imagenet mean
    transforms.Lambda(lambda x: x.mul_(255))
    ])

def load_img(path):
    img = Image.open(path)
    img = Variable(transform_img(img))
    img = img.unsqueeze(0)
    return img

def save_img(img, filename='transfer_final.png'):
    post = transforms.Compose([
         transforms.Lambda(lambda x: x.mul_(1./255)),
         transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
         transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
         ])
    img = post(img)
    img = img.clamp_(0,1)
    utils.save_image(img,
                '%s/%s' % ("./images", filename),
                normalize=True)
    return