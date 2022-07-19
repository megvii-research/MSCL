import imageio
import numpy as np
import os
import time
import torch
import cv2
import warnings
from datamaid2.utils.logconf import get_logger
from easydict import EasyDict
from torchvision import transforms
from skimage.transform import resize as imresize
from tools.ARFlow.flow_utils import flow_to_image, calc_corner_bbox_freq, \
    resize_flow, flow_to_bbox, smooth_bbox_dp, calc_nearby_bbox_freq

from .core.raft import RAFT
from .core.utils.utils import InputPadder
from .core.utils import flow_viz

logger = get_logger(__name__)
def viz(img, flo, imname='image.png'):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", imname)
    cv2.imwrite(tar, img_flo[:, :, [2,1,0]])    # imshow div 255

class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, image):
        h, w, _ = image.shape
        if h == self.new_h and w == self.new_w:
            return image
        image = imresize(image, (self.new_h, self.new_w))
        return image

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # Handle numpy array
        tensor = torch.from_numpy(array)
        # Put it from HWC to CHW format
        return tensor.float()

# The test helper for ARFlow
class TestHelper:
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        
        self.input_transform = transforms.Compose([
            Zoom(*self.cfg.test_shape),
            ArrayToTensor(),
        ])

    # Init model for ARFlow
    def init_model(self):
        model = RAFT(self.cfg.model)
        model = torch.nn.DataParallel(model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model.load_state_dict(torch.load(self.cfg.pretrained_model, map_location=self.device))

        model = model.module
        model = model.to(self.device)
        model.eval()
        return model

    # Run single image instances
    @torch.no_grad()
    def run(self, imgs, only_up=True, transform=True):
        if transform:
            imgs = [self.input_transform(img)[None].to(self.device) for img in imgs]
        else:
            imgs = [img.to(self.device) for img in imgs]
        image1, image2 = imgs
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        if only_up:
            return self.model(image1, image2, iters=20, test_mode=True)[1]
        return self.model(image1, image2, iters=20, test_mode=True)

    @torch.no_grad()
    # Run for images in a whole video sequence
    def run_sequence(self, imgs, size, gap=8, init_adjacent=8, transform=True, visualize=False):
        if transform:
            imgs = [self.input_transform(img)[None].to(self.device) for img in imgs]
        else:
            imgs = [img.to(self.device) for img in imgs]
        flows = []
        # Here introduces a trick of determining the frame interval for estimating optical flow
        # This interval is indicated as T_f in paper, and as variable 'adjacent' in the following codes
        # Basic logic: the interval for estimating flow should decline if the image is flowing too fast, and vice versa
        # We actually init the interval as 4, and it will fluctuate between 1 and 7 according to the estimated flow
        adjacent = init_adjacent
        # if adjacent != gap:
        #     print(f"adjacent {adjacent} with gap {gap} lead to {len(imgs)} \
        #         vs {len(range(0, len(imgs) - adjacent, gap))}")

        # Also note here that we actually estimate flow map and sample candidate boxes on sub-sampled videos
        # i.e. estimate flow map every 'gap' frames, by default the param 'gap' is chosen as 3
        for i in range(0, len(imgs) - adjacent, gap):
            image1, image2 = imgs[i], imgs[i+adjacent]
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            if visualize:
                viz(image1, flow_up, f"{i}.png")

            # *Following codes come from USOT
            flow_up = flow_up[0].detach().cpu().numpy().transpose([1, 2, 0])
            # TODO: support adaptively adjacent. 
            flows.append(flow_up)
        return flows

    def inference_flows(self, image_list, gap=3, init_adjacent=4, transform=True, visualize=False):

        # Load all frames in a video sequence
        imgs = image_list
        # imgs = [imageio.imread(img).astype(np.float32) for img in image_list]
        h, w = imgs[0].shape[:2]

        # Estimating optical flow for the whole video sequence
        # Note that we actually estimate flow maps and sample candidate boxes on sub-sampled videos
        # i.e. estimate flow map every 'gap' frames, by default the param 'gap' is chosen as 3 (except YT-VOS)
        flows = self.run_sequence(imgs, size=(h, w), gap=gap, init_adjacent=init_adjacent, transform=transform, visualize=visualize)
        return flows

    def inference_bboxs(self, image_list, flows, vis=True, gap=3):
        imgs = image_list
        # Note that cut_ratio is used to cut the margins of the flow map, as margin flows are always of low quality
        cut_ratio = 1/32
        # Convert flow map to candidate boxes (B)
        bboxs = [flow_to_bbox(flow, cut_ratio=cut_ratio) for flow in flows]
        # Use Dynamic Programming (DP) to generate reliable pseudo box sequences (B')
        bboxs, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary = \
            smooth_bbox_dp(bboxs, length=len(imgs), gap=gap)

        # Calc the bbox DP-select rate (bbox_freq) for every frame among all its adjacent frames
        # Note: search range is the frame interval for calculating frame quality (Denoted as T_s in the paper)
        # In practice, short interval (3) is better according to our experiments (10 is deprecated)
        freq_dict = calc_nearby_bbox_freq(picked_frame_index, video_length=len(bboxs),
                                        search_range=[3, 10], gap=gap)

        # The frequency of corner bboxes in the smoothed bbox sequence (B')
        # As an implementation detail, we actually give priority to sequences with less corner boxes (for center bias)
        corner_bbox_freq = calc_corner_bbox_freq(bboxs, img_shape=flows[0].shape, cut_ratio=cut_ratio)

        # Visualize optical flow
        # flow_vis = [flow_to_image(flow) for flow in flows]

        if vis:
            i = 0
            while True:
                if i >= len(imgs):
                    i = 0
                bbox = bboxs[i]
                image = imgs[i].astype(np.uint8)
                text = "{:.2f}/{:.2f}".format(freq_dict[i][0], freq_dict[i][1])
                draw = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                    (0, 255, 0), 1)
                draw = cv2.putText(draw, text, (int(bbox[0])+25, int(bbox[1])+25),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                tar = os.path.join("/home/nijingcheng/mmaction2/tools/RAFT/visualize", f"{i*gap}_box.png")
                cv2.imwrite(tar, draw*255)
                i += 1
                # time.sleep(0.05)
                # key = cv2.waitKey(1) & 0xFF
                # # If the `q` key was pressed, break from the loop
                # if key == ord("q"):
                #     break

        return bboxs, picked_frame_index, (freq_dict, bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq)


# Init the flow estimation module
def init_module(model=None, test_shape=None):
    if test_shape is None:
        test_shape = [384, 640]
    if model is None:
        model = "/home/nijingcheng/mmaction2/tools/RAFT/models/raft-things.pth"

    cfg = {
        'model': dict(small=False, mixed_precision=False, alternate_corr=False),
        'pretrained_model': model,
        'test_shape': test_shape,
    }
    ts = TestHelper(cfg)
    return ts

