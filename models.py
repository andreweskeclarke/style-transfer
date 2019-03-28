import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms, utils

class Vgg19(nn.Module):

    ALL_LAYERS = [
            1, 3, # Block 1
            6, 8, # Block 2
            11, 13, 15, 17, # Block 3
            20, 22, 24, 26, # Block 4
            29, 31, 33, 35 # Block 5
            ]
    # CONTENT_LAYERS = [22]
    # STYLE_LAYERS = [1, 6, 11, 20, 29]
    CONTENT_LAYERS = ALL_LAYERS[11:]
    STYLE_LAYERS = ALL_LAYERS[:11]

    def __init__(self, indexes):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        vgg.cuda()
        n_features = min(len(vgg.features), max(indexes))
        features = list(vgg.features)[:n_features+1]
        self.features = nn.ModuleList(features).eval()
        self.indexes = sorted(set(indexes))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, output_indexes=None):
        results = {}
        for index, model in enumerate(self.features):
            x = model(x)
            if index in self.indexes:
                results[index] = x
        if output_indexes is not None:
            return list([results[i] for i in output_indexes])
        return list([results[i] for i in self.indexes])
