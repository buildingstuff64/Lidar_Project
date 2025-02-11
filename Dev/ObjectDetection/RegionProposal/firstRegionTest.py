import torch
import torch.nn as nn
import torchvison
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        #3*3 conv
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride=1, padding=1)

        #1*1 classifaction
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size = 1, stride=1)

        #1*1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.nn_anchors * 4, kernel_size = 1, stride=1)

    def forward(self, image, feat, target):
        #call rpn layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        boc_transform_pred = self.bbox_reg_layer(rpn_feat)

        #generate anchors
        anchors = self.generate_anchors(image, feat)

    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype = torch.int64, device = feat.device)

        scales = torch.as_tensor(self.scales, dtype = feat.dtype, device =feat.device)
        aspect_ratios =torch.as_tensor(self.aspect_ratios, dtype = feat.dtype, device = feat.device)

        #below
        h_ratios =torch.sqrt(aspect_ratios)
        w_ratios = 1/h_ratios

        ws = (w_ratios[:,None] * scales[None, :]).view(-1)
        hs = (h_ratios[:,None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim = 1)/2
        base_anchors = base_anchors.round()
