from RAFT.inference import init_module
import refile
import nori2 as nori
import numpy as np
import time
import torch
from meghair.utils import io
from meghair.utils.imgproc import imdecode
fetcher = nori.Fetcher()

ts = init_module()
path = "s3://activity-public/kinetics600/annos/kinetics155_temporal_val.pkl"
with refile.smart_load_from(path) as f:
    annos = io.load(f)
anno = annos[0]
nori_id_seq = anno['nori_id_seq']
frames = [imdecode(fetcher.get(nid)) for nid in nori_id_seq]
frames = [frame.astype(np.float32)[..., ::-1].copy() for frame in frames]
st = time.time()
gap = 8
with torch.no_grad():
    flows = ts.inference_flows(frames, gap=gap, init_adjacent=8, visualize=True)
    ts.inference_bboxs(frames[::gap], flows, vis=True, gap=1)
print(time.time()-st)
print(type(flows))
print(type(flows[0]))
print(flows[0].shape, flows[0].max(), flows[0].min(), flows[0].mean())