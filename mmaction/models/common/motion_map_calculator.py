import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _forward_unimplemented

class MotionMapCalculator(nn.Module):
    def __init__(self, scales=(7, 7), pool_type='max'):
        # (7, 7) for 112 input
        super().__init__()
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = sobel_kernel_x.T

        self.sobel_filter_x = nn.Conv2d(1, 1, 3, padding=1, bias=False, padding_mode='replicate')
        self.sobel_filter_x.weight.data = sobel_kernel_x.unsqueeze(0).unsqueeze(0)
        self.sobel_filter_x.weight.requires_grad = False
        self.sobel_filter_y = nn.Conv2d(1, 1, 3, padding=1, bias=False, padding_mode='replicate')
        self.sobel_filter_y.weight.data = sobel_kernel_y.unsqueeze(0).unsqueeze(0)
        self.sobel_filter_y.weight.requires_grad = False
        self.pool_type = pool_type
        self.scales = scales

    def forward(self, flows):
        # flows: b,2,t,h,w
        bs, _, t = flows.shape[:3]
        u, v = torch.chunk(flows, 2, dim=1) # b,1,t,h,w
        u, v = u[:, 0].flatten(0, 1), v[:, 0].flatten(0, 1)
        u, v = u[:, None], v[:, None]   # bt,1,h,w

        u_x = self.sobel_filter_x(u)
        u_y = self.sobel_filter_y(u)
        v_x = self.sobel_filter_x(v)
        v_y = self.sobel_filter_y(v)
        motion_map = torch.sqrt(u_x**2 + u_y**2 + v_x**2 + v_y**2)      # 边缘图
        if self.pool_type == 'max':
            motion_map = F.max_pool2d(motion_map, kernel_size=self.scales, stride=self.scales)
        elif self.pool_type == 'avg':
            motion_map = F.avg_pool2d(motion_map, kernel_size=self.scales, stride=self.scales)
        else:
            raise ValueError(f"Not support {self.pool_type}")
        motion_map = F.interpolate(motion_map, scale_factor=self.scales, mode='bilinear')       # coarsen的边缘图
        assert motion_map.requires_grad is False
        assert motion_map.shape[-2:] == flows.shape[-2:]
        motion_map = motion_map.unflatten(0, (bs, t)).squeeze(2)    # b,t,h,w
        return motion_map