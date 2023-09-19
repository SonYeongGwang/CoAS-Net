import torch
import torch.nn.functional as F
import copy

from torch import conv2d, nn

from modeling import FeaturePyramidNetwork, resnet34, resnet50, resnet101
from collections import OrderedDict

def load_feature_backbone(backbone_name, pretrained_backbone):
    if backbone_name=="resnet34":
        rgb_backbone = resnet34(pretrained=pretrained_backbone, progress=True)

    elif backbone_name=="resnet50":
        rgb_backbone = resnet50(pretrained=pretrained_backbone, progress=True)

    elif backbone_name=="resnet101":
        rgb_backbone = resnet101(pretrained=pretrained_backbone, progress=True)

    depth_backbone = copy.deepcopy(rgb_backbone)

    return rgb_backbone, depth_backbone

def combine_pyramid_feats(pyramid_feats, upsample_size=(120, 160)):
        p = []
        keys = pyramid_feats.keys()
        for key in keys:
            feat = pyramid_feats[key]
            p.append(F.interpolate(feat, upsample_size, mode='bilinear', align_corners=False))
        
        combined_feats = torch.cat(p, dim=1)
        return combined_feats

class FullyConvFeatureFuseModel(nn.Module):
    def __init__(self, backbone_name, pretrained_backbone):
        super(FullyConvFeatureFuseModel, self).__init__()
        self.rgb_backbone, self.depth_backbone = load_feature_backbone(backbone_name, pretrained_backbone)

        if backbone_name == "resnet34":
            self.n_feature_maps = 512

        elif backbone_name == "resnet50":
            self.n_feature_maps = 2048
        
        elif backbone_name == "resnet101":
            self.n_feature_maps = 2048

        self.rgb_pyramid = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        self.depth_pyramid = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)

        self.rgb_merge_conv = nn.Conv2d(256*4, 256, (3, 3), padding=1)
        self.depth_merge_conv = nn.Conv2d(256*4, 256, (3, 3), padding=1)

    def forward(self, x1, x2):

        rgb_staged_feat = self.rgb_backbone(x1)
        depth_staged_feat = self.depth_backbone(x2)

        rgb_pyramid_feat = self.rgb_pyramid(rgb_staged_feat)
        depth_pyramid_feat = self.depth_pyramid(depth_staged_feat)

        rgb_feats = combine_pyramid_feats(rgb_pyramid_feat, (120, 160))
        depth_feats = combine_pyramid_feats(depth_pyramid_feat, (120, 160))

        dense_rgb_feats = self.rgb_merge_conv(rgb_feats)
        dense_depth_feats = self.depth_merge_conv(depth_feats)
        
        fused_features = torch.cat((dense_rgb_feats, dense_depth_feats), dim=1)
        return fused_features

class SuctionModel(nn.Module):
    def __init__(self, backbone_name, pretrained_backbone):
        super(SuctionModel, self).__init__()
        self.__output_size = (480, 640)
        self.input = input

        self.m = nn.Softmax(dim=1)

        self.fuse_model = FullyConvFeatureFuseModel(backbone_name, pretrained_backbone)
        self.detector = nn.Sequential(OrderedDict([
                                                ('conv1', nn.Conv2d(512, 256, (1, 1))),
                                                ('conv2', nn.Conv2d(256, 128, (1, 1))),
                                                ('conv3', nn.Conv2d(128, 3, (1, 1)))
                                                ]))

        # self.detector = nn.Conv2d(512, 3, (1, 1))
        for m in self.fuse_model.rgb_merge_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.fuse_model.depth_merge_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.detector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def size(self) -> tuple:
        # print("insize the getter")
        return self.__output_size

    @size.setter
    def size(self, size: tuple):
        # print("insize the setter")
        self.__output_size = size

    def forward(self, x1, x2=None):

        out = self.fuse_model(x1, x2)

        out = self.detector(out)
        ## when 3x480x640 images are given, the output size of the last layer before upsampling is (2, 15, 20) 
        out = F.interpolate(out, self.__output_size, mode='bilinear', align_corners=False)
        out = self.m(out)

        return out