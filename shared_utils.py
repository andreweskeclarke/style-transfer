# Copyright (c) 2018 Pawan Sasanka Ammanamanchi and licensed under the MIT LIcense
# Available at https://github.com/Shashi456/Neural-Style

import torch
import os
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
from PIL import Image

#Dataset Processing
transform_img = transforms.Compose([
    transforms.Resize(512), #Default image_size
    transforms.ToTensor(), #Transform it to a torch tensor
    transforms.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    #transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]), #subracting imagenet mean
    transforms.Lambda(lambda x: x.mul_(255))
    ])

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def load_img(path):
    img = Image.open(path)
    img = Variable(transform_img(img), requires_grad=False)
    img = img.unsqueeze(0)
    return img

def save_img(img, filename):
    img = img.data[0].cpu().squeeze()
    post = transforms.Compose([
         transforms.Lambda(lambda x: x.mul_(1./255)),
         #transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
         transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
         ])
    img = post(img)
    img = img.clamp_(0,1)
    image_filepath = os.path.join(filename)
    os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
    utils.save_image(img, image_filepath, normalize=True)
    return image_filepath

def merge_images(paths, output_path):
    images = list([Image.open(p) for p in paths])
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(output_path)
