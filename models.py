import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms, utils

class Vgg19(nn.Module):
    def __init__(self, indexes):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        vgg.cuda()
        features = list(vgg.features)[:37]
        self.features = nn.ModuleList(features).eval()
        self.indexes = sorted(set(indexes))

    def forward(self, x, output_indexes=None):
        results = {}
        for index, model in enumerate(self.features):
            x = model(x)
            if index in self.indexes:
                results[index] = x
        if output_indexes is not None:
            return list([results[i] for i in output_indexes])
        return list([results[i] for i in self.indexes])
