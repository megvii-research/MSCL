import torch.nn as nn

from torchvision.models.utils import load_state_dict_from_url

from mmaction.models.recognizers.moco import forward


__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride

class Conv3DNoDownSample(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoDownSample, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(padding, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 16, kernel_size=(1, 7, 7), stride=(2, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

class BasicStem2D(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem2D, self).__init__(
            nn.Conv3d(6, 16, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = x.unflatten(2, (x.shape[2]//2, 2))
        # 这里实际上是: r0,r1,g0,g1,b0,b1
        x = x.transpose(2, 3).flatten(1, 2) # n,2c,t,h,w
        return super().forward(x)

class BasicStem2Dv2(nn.Sequential):
    """2Dv2: Sample with stride 2
    """
    def __init__(self):
        super(BasicStem2Dv2, self).__init__(
            nn.Conv3d(3, 16, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = x[:, :, ::2]
        return super().forward(x)

class BottleneckStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    This only used for flow encoding!!
    """
    def __init__(self):
        super(BottleneckStem, self).__init__(
            nn.Conv3d(3, 8, kernel_size=(1, 7, 7), stride=(2, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        if issubclass(block, BasicBlock):
            self.inplanes = 16
        elif issubclass(block, Bottleneck):
            self.inplanes = 8
        else:
            raise ValueError(f"Tpye: {block}")
        base_inplanes = self.inplanes

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], base_inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], base_inplanes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], base_inplanes*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], base_inplanes*8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(8*base_inplanes*block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def r3dv2_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DNoDownSample] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def mx2d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('mx2d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DNoTemporal] * 3 + [Conv3DSimple],
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def r2d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('r2d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DNoTemporal] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)
                         

def r2dv2_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network, v2 -> 6 channels
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('r2d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DNoTemporal] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem2D, **kwargs)

def r2dv3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network, v3 -> strides 2
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('r2d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DNoTemporal] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem2Dv2, **kwargs)

def r2d_50(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer 2D Convolution network
    """
    assert pretrained is False, "Not support pretrain for r2d_18."
    return _video_resnet('r2d_50',
                         pretrained, progress,
                         block=Bottleneck,
                         conv_makers=[Conv3DNoTemporal] * 4,
                         layers=[3, 4, 6, 3],
                         stem=BottleneckStem, **kwargs)

def ResNetFlow(name, pretrained=False, disable_clf=False, **kwargs):
    if name == "r2d_18":
        backbone = r2d_18(pretrained=pretrained, **kwargs)
    elif name == "r2dv2_18":
        backbone = r2dv2_18(pretrained=pretrained, **kwargs)
    elif name == "r2dv3_18":
        backbone = r2dv3_18(pretrained=pretrained, **kwargs)
    elif name == "mx2d_18":
        backbone = mx2d_18(pretrained=pretrained, **kwargs)
    elif name == "r2d_50":
        backbone = r2d_50(pretrained=pretrained, **kwargs)
    elif name == "mc3_18":
        backbone = mc3_18(pretrained=pretrained, **kwargs)
    elif name == "r3d_18":
        backbone = r3d_18(pretrained=pretrained, **kwargs)
    elif name == "r3dv2_18":
        backbone = r3dv2_18(pretrained=pretrained, **kwargs)
    else:
        raise NotImplementedError(f"{name}")
    if disable_clf:
        backbone.classifier = nn.Identity()
        backbone.fc = nn.Identity()
    return backbone