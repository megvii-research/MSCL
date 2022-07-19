import numpy as np

from ..builder import PIPELINES
from tools.RAFT.core.utils import flow_viz
# cv2.setNumThreads(0)

def norm_flow(flow_uv, clip_flow=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return np.stack((u, v), axis=-1)


@PIPELINES.register_module()
class Flow2ImgWithAug(object):
    def __init__(self, ratios, convert_to_bgr=False, merge_aug=True):
        self.ratios = ratios    # e.g. (0.2, 1) -> (0.2pi, pi)
        self.convert_to_bgr = convert_to_bgr
        self.merge_aug = merge_aug

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []
        rotated_flow_imgs = []

        for flow_uv in flow_uv_list:
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]

            beta = np.random.uniform(*self.ratios)*np.pi
            sin_beta, cos_beta = np.sin(beta), np.cos(beta)
            new_u = cos_beta*u - sin_beta*v
            new_v = sin_beta*u + cos_beta*v
            new_flow_uv = np.stack((new_u, new_v), axis=-1)

            flow_imgs.append(flow_viz.flow_to_image(flow_uv, convert_to_bgr=self.convert_to_bgr))
            rotated_flow_imgs.append(flow_viz.flow_to_image(new_flow_uv, convert_to_bgr=self.convert_to_bgr))
        if self.merge_aug:
            results['flow_imgs'] = flow_imgs + rotated_flow_imgs
        else:
            results['flow_imgs'] = flow_imgs
            results['rotated_flow_imgs'] = rotated_flow_imgs

        return results


@PIPELINES.register_module()
class NormFlowWithAug(object):
    def __init__(self, ratios, merge_aug=True):
        self.ratios = ratios    # e.g. (0.2, 1) -> (0.2pi, pi)
        self.merge_aug = merge_aug

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []
        rotated_flow_imgs = []

        beta = np.random.uniform(*self.ratios)*np.pi
        sin_beta, cos_beta = np.sin(beta), np.cos(beta)
        for flow_uv in flow_uv_list:
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]

            
            new_u = cos_beta*u - sin_beta*v
            new_v = sin_beta*u + cos_beta*v
            new_flow_uv = np.stack((new_u, new_v), axis=-1)

            flow_imgs.append(norm_flow(flow_uv))
            rotated_flow_imgs.append(norm_flow(new_flow_uv))
        if self.merge_aug:
            results['flow_imgs'] = flow_imgs + rotated_flow_imgs
        else:
            results['flow_imgs'] = flow_imgs
            results['rotated_flow_imgs'] = rotated_flow_imgs
        del results['flows']

        return results


@PIPELINES.register_module()
class NormFlowWithStidedAug(object):
    def __init__(self, ratios, num_chunks, merge_aug=True):
        self.ratios = ratios    
        self.start = ratios[0]  # e.g. 0.2
        self.stride = (ratios[1]-ratios[0])/num_chunks
        self.num_chunks = num_chunks
        self.merge_aug = merge_aug

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []
        rotated_flow_imgs = []

        cid = np.random.randint(0, self.num_chunks)
        for flow_uv in flow_uv_list:
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]

            
            beta = (self.start + self.stride*cid)*np.pi
            sin_beta, cos_beta = np.sin(beta), np.cos(beta)
            new_u = cos_beta*u - sin_beta*v
            new_v = sin_beta*u + cos_beta*v
            new_flow_uv = np.stack((new_u, new_v), axis=-1)

            flow_imgs.append(norm_flow(flow_uv))
            rotated_flow_imgs.append(norm_flow(new_flow_uv))
        aug_labels = cid
        if self.merge_aug:
            results['flow_imgs'] = flow_imgs + rotated_flow_imgs
        else:
            results['flow_imgs'] = flow_imgs
            results['rotated_flow_imgs'] = rotated_flow_imgs
        results['ap_labels'] = aug_labels      # start from 0
        del results['flows']

        return results


@PIPELINES.register_module()
class NormFlowWithStidedAugV2(object):
    """
    v2: Similar with v1, but use flows rather than flow_imgs as output key.
    """
    def __init__(self, ratios, num_chunks, merge_aug=True):
        self.ratios = ratios    
        self.start = ratios[0]  # e.g. 0.2
        self.stride = (ratios[1]-ratios[0])/num_chunks
        self.num_chunks = num_chunks
        self.merge_aug = merge_aug

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []
        rotated_flow_imgs = []

        cid = np.random.randint(0, self.num_chunks)
        for flow_uv in flow_uv_list:
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]

            
            beta = (self.start + self.stride*cid)*np.pi
            sin_beta, cos_beta = np.sin(beta), np.cos(beta)
            new_u = cos_beta*u - sin_beta*v
            new_v = sin_beta*u + cos_beta*v
            new_flow_uv = np.stack((new_u, new_v), axis=-1)

            flow_imgs.append(norm_flow(flow_uv))
            rotated_flow_imgs.append(norm_flow(new_flow_uv))
        aug_labels = cid
        if self.merge_aug:
            results['flows'] = flow_imgs + rotated_flow_imgs
        else:
            results['flows'] = flow_imgs
            results['rotated_flows'] = rotated_flow_imgs
        results['ap_labels'] = aug_labels      # start from 0

        return results


@PIPELINES.register_module()
class NormFlowV2(object):
    """
    v2: Similar with v1, but use flows rather than flow_imgs as output key.
    """
    def __init__(self):
        pass

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []

        for flow_uv in flow_uv_list:
            flow_imgs.append(norm_flow(flow_uv))
        results['flows'] = flow_imgs
        return results


@PIPELINES.register_module()
class NormFlowWithAugV2(object):
    """
    v2: Similar with v1, but use flows rather than flow_imgs as output key.
    """
    def __init__(self, ratios, merge_aug=True):
        self.ratios = ratios    # e.g. (0.2, 1) -> (0.2pi, pi)
        self.merge_aug = merge_aug

    def __call__(self, results):
        # cal ToTensor before
        # crop the input frames from raw clip
        flow_uv_list = results['flows']    # ndarray, h,w,2
        flow_imgs = []
        rotated_flow_imgs = []

        beta = np.random.uniform(*self.ratios)*np.pi
        sin_beta, cos_beta = np.sin(beta), np.cos(beta)
        for flow_uv in flow_uv_list:
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]
            
            new_u = cos_beta*u - sin_beta*v
            new_v = sin_beta*u + cos_beta*v
            new_flow_uv = np.stack((new_u, new_v), axis=-1)

            flow_imgs.append(norm_flow(flow_uv))
            rotated_flow_imgs.append(norm_flow(new_flow_uv))
        if self.merge_aug:
            results['flows'] = flow_imgs + rotated_flow_imgs
        else:
            results['flows'] = flow_imgs
            results['rotated_flows'] = rotated_flow_imgs

        return results