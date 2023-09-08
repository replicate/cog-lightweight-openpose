import collections

import torch
from torch import nn


def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for k, v in target_state.items():
        if k in source_state and source_state[k].size() == target_state[k].size():
            new_target_state[k] = source_state[k]
        else:
            new_target_state[k] = target_state[k]
            print('[WARNING] Not found pre-trained parameters for {}'.format(k))

    net.load_state_dict(new_target_state)

def conv(
    in_channels, 
    out_channels, 
    kernel_size=3, 
    padding=1, 
    batchnorm=True, 
    dilation=1, 
    stride=1, 
    relu=True, 
    bias=True
):
    modules = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
    ]
    if batchnorm:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))

    return nn.Sequential(*modules)


def conv_depthwise(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=False
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_depthwise_no_batchnorm(
    in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=False
        ),
        nn.ELU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, batchnorm=False)
        self.trunk = nn.Sequential(
            conv_depthwise_no_batchnorm(out_channels, out_channels),
            conv_depthwise_no_batchnorm(out_channels, out_channels),
            conv_depthwise_no_batchnorm(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, batchnorm=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, batchnorm=False),
            conv(num_channels, num_channels, batchnorm=False),
            conv(num_channels, num_channels, batchnorm=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, batchnorm=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, batchnorm=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, batchnorm=False),
            conv(512, num_pafs, kernel_size=1, padding=0, batchnorm=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, batchnorm=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, batchnorm=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, batchnorm=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, batchnorm=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, batchnorm=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_depthwise( 32,  64),
            conv_depthwise( 64, 128, stride=2),
            conv_depthwise(128, 128),
            conv_depthwise(128, 256, stride=2),
            conv_depthwise(256, 256),
            conv_depthwise(256, 512),  # conv4_2
            conv_depthwise(512, 512, dilation=2, padding=2),
            conv_depthwise(512, 512),
            conv_depthwise(512, 512),
            conv_depthwise(512, 512),
            conv_depthwise(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()

        for idx in range(num_refinement_stages):
            self.refinement_stages.append(
                RefinementStage(
                    num_channels + num_heatmaps + num_pafs, 
                    num_channels,
                    num_heatmaps, 
                    num_pafs
                )
            )

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            output = torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)
            stages_output.extend(refinement_stage(output))

        return stages_output
