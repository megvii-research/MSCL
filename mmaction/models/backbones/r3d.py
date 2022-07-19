from inspect import getblock
import torch.nn as nn

from torchvision.models.utils import load_state_dict_from_url
from mmcv.utils import _BatchNorm

from ..builder import BACKBONES
__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

try:
    from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1,
                 dilation=(1, 1, 1),):
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(stride[0], stride[1], stride[2]),
            padding=padding,
            dilation=dilation,
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
                 padding=1,
                 dilation=(1, 1, 1),):
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        super(Conv2Plus1D, self).__init__(
            # dilation only infulence spatial
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride[1], stride[2]), padding=(0, padding, padding),
                      dilation=dilation, bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride[0], 1, 1), padding=(padding, 0, 0),
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
                 padding=1,
                 dilation=(1, 1, 1),):
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding, padding),
            dilation=dilation,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, dilation=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride, dilation=dilation),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, dilation=dilation),
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

    def __init__(self, inplanes, planes, conv_builder, stride=1, dilation=1, downsample=None):

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
            conv_builder(planes, planes, midplanes, stride, dilation=dilation),
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
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class BasicDownSampleStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=1))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


@BACKBONES.register_module()
class R3D(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, use_dilation=False, num_classes=400,
                 zero_init_residual=False, frozen_stages=-1, **kwargs):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
            frozen_stages: -1, 0, n, 0 means only stem, n means front n layers
        """
        super().__init__()
        block = get_block_from_name(block)
        if isinstance(conv_makers, str):
            conv_makers = [get_conv_maker_from_name(conv_makers)]*4
        else:
            conv_makers = [get_conv_maker_from_name(conv_maker) for conv_maker in conv_makers]
        stem = get_stem_from_name(stem)
        
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, use_dilation=use_dilation)

        # init weights
        self._initialize_weights()
        self.frozen_stages = frozen_stages

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

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, use_dilation=False):
        downsample = None

        dilation = (1, 2, 2) if use_dilation else 1
        ds_stride = stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            if use_dilation:
                # Replace downsample
                ds_stride = (ds_stride[0], 1, 1)
                dilation = (1, ds_stride[1], ds_stride[2])
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=ds_stride,
                            dilation=dilation, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=ds_stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, ds_stride, dilation=dilation, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, dilation=dilation))

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
    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.stem.eval()    # Sequential will eval bn iterately
            for m in self.stem.modules():
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         if isinstance(m, _BatchNorm):
        #             m.eval()
        # if mode and self.partial_bn:
        #     self._partial_bn()

def get_block_from_name(name):
    if name == 'BasicBlock':
        return BasicBlock
    elif name == 'Bottleneck':
        return Bottleneck
    else:
        raise ValueError(name)

def get_stem_from_name(name):
    if name == 'BasicStem':
        return BasicStem
    elif name == 'R2Plus1dStem':
        return R2Plus1dStem
    elif name == 'BasicDownSampleStem':
        return BasicDownSampleStem
    else:
        raise ValueError(name)

def get_conv_maker_from_name(name):
    if name == 'Conv3DSimple':
        return Conv3DSimple
    elif name == 'Conv3DNoTemporal':
        return Conv3DNoTemporal
    elif name == 'Conv2Plus1D':
        return Conv2Plus1D
    else:
        raise ValueError(name)

if mmdet_imported:
    MMDET_BACKBONES.register_module()(R3D)