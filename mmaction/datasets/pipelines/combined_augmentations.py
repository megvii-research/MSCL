from torchvision import transforms as transforms
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import _functional_video as F

from .transforms_torch import build_torch_transforms
from ..builder import PIPELINES
from torch.utils.data.dataloader import default_collate

class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode), (i, j, h, w)

    def apply_transform(self, clip, i, j, h, w):
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def apply_transform_with_scale(self, clip, i, j, h, w, h_rate, w_rate):
        i, j, h, w = int(round(i*h_rate)), int(round(j*w_rate)), int(round(h*h_rate)), int(round(w*w_rate))
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation_mode={1}, scale={2}, ratio={3})".format(
            self.size, self.interpolation_mode, self.scale, self.ratio
        )


@PIPELINES.register_module()
class MoCoTransform:
    """Remove flip from ResimTransform
    """

    def __init__(
        self, transform=None, crop_transform=None, ending_transform=None,
        img_size=224, flow_key="flows",
    ):
        self.base_transform = build_torch_transforms(transform)
        self.crop_transform = RandomResizedCropVideo(**crop_transform)
        self.ending_transform = build_torch_transforms(ending_transform)

        self.anchor_rate = img_size/224
        self.flow_key = flow_key

    def __call__(self, results):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgs = results["imgs"]
        clip_len = results["clip_len"]
        # imgs = torch.from_numpy(np.array(imgs, dtype=np.uint8))
        imgs = default_collate(imgs)        # t, h, w, c
        h, w = imgs.shape[-3], imgs.shape[-2]

        if clip_len == imgs.shape[0]:
            imgs_q, imgs_k = imgs, imgs
        else:
            imgs_q, imgs_k = imgs.chunk(2, 0)
        # assert clip_len == imgs.shape[0], f"{clip_len} vs {imgs.shape[0]}, \
        #     repeat_num=1 is needed!"

        # base transform (might be empty)
        if self.base_transform is not None:
            q, k = self.base_transform(imgs_q), self.base_transform(imgs_k)
        else:
            q, k = imgs_q, imgs_k

        q, (t_q, l_q, h_q, w_q) = self.crop_transform(q)
        k, (t_k, l_k, h_k, w_k) = self.crop_transform(k)
        if self.flow_key in results:
            # Only support flow image
            flows = results[self.flow_key]      # list of ndarray (h, w, c)
            flows = default_collate(flows)
            h_f, w_f = flows.shape[-3], flows.shape[-2]
            if clip_len == flows.shape[0]:
                flows_q, flows_k = flows, flows
            else:
                flows_q, flows_k = flows.chunk(2, 0)
            # Base transform
            if self.base_transform is not None:
                flows_q, flows_k = self.base_transform(flows_q), self.base_transform(flows_k)
            h_rate, w_rate = h_f/h, w_f/w
            flows_q = self.crop_transform.apply_transform_with_scale(flows_q, t_q, l_q, h_q, w_q, h_rate, w_rate)
            flows_k = self.crop_transform.apply_transform_with_scale(flows_k, t_k, l_k, h_k, w_k, h_rate, w_rate)

            results[self.flow_key] = [flows_q, flows_k]

        q, k = self.ending_transform(q), self.ending_transform(k)

        results["imgs"] = [q, k]
        return results