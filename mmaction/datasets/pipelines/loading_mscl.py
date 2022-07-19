import numpy as np

from ..builder import PIPELINES
from .loading import SampleFrames

import warnings

@PIPELINES.register_module()
class AlignIndex:
    """Align index for img and flow.
    """
    def __init__(self, gap, adjacent) -> None:
        self.gap = gap
        self.adjacent = adjacent

    def __call__(self, results):
        len_flow = None
        if "nids_flow" in results:
            len_flow = len(results["nids_flow"])
        if "nids_flow_img" in results:
            if len_flow is not None:
                assert len(results["nids_flow_img"]) == len_flow
            else:
                len_flow = len(results["nids_flow_img"])
        if "gt_bboxes" in results:
            assert len(results["gt_bboxes"]) == len_flow
        nori_id_seq = results["nori_id_seq"]
        len_img = len(nori_id_seq)
        assert len_flow == (len_img-self.adjacent)//self.gap, f"{len_flow} vs {len_img}"
        nori_id_seq = nori_id_seq[0:len_img-self.adjacent:self.gap]
        total_frames = len(nori_id_seq)
        assert total_frames == len_flow, f"{total_frames} vs {len_flow}"
        results["nori_id_seq"] = nori_id_seq
        results["total_frames"] = total_frames
        return results


@PIPELINES.register_module()
class FlowToGT:
    """Only bboxs generated by flow is scaled.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, results):
        flow_prposals = results["flow_prposals"]
        if "gt_bboxes" in results:
            raise NotImplementedError("Not support now!")
        results["gt_bboxes"] = flow_prposals
        

@PIPELINES.register_module()
class MatchFlow:
    """Sample frames to flow size before sampler.
    """
    def __init__(self, gap=2, adjacent=8, flow_key="nids_flow_img") -> None:
        self.gap = gap
        self.adjacent = adjacent
        self.flow_key = flow_key

    def __call__(self, results):
        nori_id_seq = results["nori_id_seq"]
        new_nori_id_seq = [nori_id_seq[idx] for idx in range(0, len(nori_id_seq) - self.adjacent, self.gap)]
        results["nori_id_seq"] = new_nori_id_seq
        results["total_frames"] = len(new_nori_id_seq)

        assert len(new_nori_id_seq) == len(results[self.flow_key]), \
            f"{len(new_nori_id_seq)} vs {len(results[self.flow_key])}"
        return results


@PIPELINES.register_module()
class Seg2T:
    def __init__(self):
        pass

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        assert clip_len == 1, "Only support one frame per clip now!"
        
        imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
        # N_crops x N_clips x L x H x W x C
        imgs = np.transpose(imgs, (0, 2, 5, 1, 3, 4))
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        results['num_clips'] = clip_len
        results['clip_len'] = num_clips

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ChosenSampleFrames(SampleFrames):
    def _sample_clips(self, num_frames, chosen_idx):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            raise NotImplementedError("Not support test mode now")
        else:
            cur_attempt = 0
            while True:
                clip_offsets = self._get_train_clips(num_frames)
                assert clip_offsets.shape[0] == 1, f"{clip_offsets.shape}"
                if clip_offsets[0] in chosen_idx:
                    break
                cur_attempt += 1
                if cur_attempt > 10:
                    cur_offset = chosen_idx[0] if len(chosen_idx) else 0
                    clip_offsets = np.array([cur_offset], dtype=np.int)    # video is too short
                    break

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        chosen_idx = results['chosen_idx']

        clip_offsets = self._sample_clips(total_frames, chosen_idx)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class TemporalShiftChosenSampleFrames(SampleFrames):
    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False,
                 shift_range=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.shift_range = shift_range*clip_len*frame_interval          # !!! In old version, frame_interval is missed
        assert self.out_of_bound_opt in ['loop', 'repeat_last']
        assert self.num_clips == 1

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _sample_clips(self, num_frames, chosen_idx):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            raise NotImplementedError("Not support test mode now")
        else:
            cur_attempt = 0
            while True:
                clip_offsets = self._get_train_clips(num_frames)
                assert clip_offsets.shape[0] == 1, f"{clip_offsets.shape}"
                if clip_offsets[0] in chosen_idx:
                    break
                cur_attempt += 1
                if cur_attempt > 10:
                    cur_offset = chosen_idx[0] if len(chosen_idx) else 0
                    clip_offsets = np.array([cur_offset], dtype=np.int)    # video is too short
                    break
        cur_shift = np.random.randint(-self.shift_range, self.shift_range+1)
        tar_offset = clip_offsets[0] + cur_shift
        new_offset = 0
        if len(chosen_idx):
            for cid in chosen_idx:
                if abs(cid-tar_offset) < abs(cid-new_offset):
                    new_offset = cid
        new_offsets = np.array([new_offset], dtype=np.int)
        clip_offsets = np.concatenate((clip_offsets, new_offsets), axis=0)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        chosen_idx = results['chosen_idx']

        clip_offsets = self._sample_clips(total_frames, chosen_idx)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips*2
        return results


@PIPELINES.register_module()
class TemporalShiftSampleFrames(SampleFrames):
    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False,
                 shift_range=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.shift_range = shift_range*clip_len*frame_interval
        assert self.out_of_bound_opt in ['loop', 'repeat_last']
        assert self.num_clips == 1

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        cur_shift = np.random.randint(-self.shift_range, self.shift_range+1)
        tar_offset = clip_offsets[0] + cur_shift
        new_offset = max(min(tar_offset, num_frames-(self.clip_len*self.frame_interval)), 0)
        new_offsets = np.array([new_offset], dtype=np.int)
        clip_offsets = np.concatenate((clip_offsets, new_offsets), axis=0)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips*2
        return results