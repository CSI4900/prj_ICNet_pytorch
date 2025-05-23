"""Base Model for Semantic Segmentation"""
import torch.nn as nn
import yaml
import os
from .base_models.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s

__all__ = ['SegBaseModel']

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, backbone='resnet50', pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = True
        self.nclass = nclass

        # Get the absolute path of the current script `segbase.py`
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(BASE_DIR, "../configs/icnet.yaml")

        # Open the file safely
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as yaml_file:
            cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, config=cfg, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, config=cfg, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, config=cfg, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        
        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        return pred
