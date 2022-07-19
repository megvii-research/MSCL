import torch
from torchvision import transforms as transforms
from torchvision.transforms import _transforms_video as transforms_video
from torchvision.transforms import Compose
from torchvision.transforms import _functional_video as Fv

class VideoResize:
    def __init__(self, size, interpolation_mode = "bilinear") -> None:
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return Fv.resize(clip, self.size, self.interpolation_mode)

name2transforms = dict(
    ToTensorVideo=transforms_video.ToTensorVideo,
    VideoResize=VideoResize,
    RandomResizedCropVideo=transforms_video.RandomResizedCropVideo,
)

def build_torch_transforms(tfms):
    new_tfms = []
    for transform in tfms:
        if isinstance(transform, dict):
            args = transform.copy()
            type = args.pop('type')
            new_tfms.append(name2transforms[type](**args))
        elif callable(transform):
            new_tfms.append(transform)
        else:
            raise TypeError(f"transform must be callable or a dict, " f"but got {type(transform)}")
        
    transforms = Compose(new_tfms) if len(new_tfms) else torch.nn.Identity()
    return transforms