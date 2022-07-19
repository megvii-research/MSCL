import argparse
import math
import os
import pickle
import subprocess
import threading

import boto3
import nori2
from datamaid2.storage.oss import OSS
from meghair.utils import io
from tqdm import tqdm

oss = OSS()

# nori_paths = []
# host = "http://oss.hh-b.brainpp.cn"
# # Client初始化
# s3_client = boto3.client('s3', endpoint_url=host)

class SpeedupThread(threading.Thread):
    def __init__(self, nori_paths):
        super(SpeedupThread, self).__init__()
        self.nori_paths = nori_paths

    def run(self):
        for path in self.nori_paths:
            try:
                ## python 初始化
                #nr = nori2.open(path)
                # nori2.speedup.on(nr)
                # nr.close()

                print('speed up : {}'.format(path))
                ## 命令行初始化
                cmd = ['nori', 'speedup', path, '--on', '--replica=1']
                subprocess.run(cmd)
            except Exception as E:
                print(E)


def get_nori_paths(args):
    if args.prefix:
        if not args.prefix.endswith('/'):
            args.prefix += '/'

        # iterdirs = list(oss.iterdir(prefix=args.prefix, full_path=True))
        iterfiles = list(oss.list_objects(prefix=args.prefix, full_path=True))
        nori_paths = set()
        for x in iterfiles:
            nori_path = os.path.dirname(x)
            if nori_path.endswith('.nori'):
                nori_path += '/'
                nori_paths.update([nori_path])
        nori_paths = list(nori_paths)
    
    elif args.pkl_path:
        nori_paths = io.load(args.pkl_path)
        nori_paths = [x if x.endswith('/') else x+'/' for x in nori_paths]
    
    else:
        raise ValueError('args.prefix or args.pkl_path is not defined!')

    return nori_paths


def speedup(args):
    # prefix = args.prefix
    # pkl_path = args.pkl_path
    grep = args.include if args.include else ['']
    exclude = args.exclude if args.exclude else None
    # paths = []
    # marker = ""
    # cnt = 0

    print('start')
    nori_paths = get_nori_paths(args)

    results = []
    for x in nori_paths:
        if not x.endswith('.nori/'):
            continue

        valid = True
        for g in grep:
            if g not in x:
                valid = False
        if exclude:
            for e in exclude:
                if e in x:
                    valid = False
        if valid:
            results.append(x)

    threads = []
    nori_paths = list(set(results))
    print('start to speedup noris, nori package number: {}'.format(len(nori_paths)))

    thread_num = min(args.thread_num, len(nori_paths))
    inter = math.ceil(len(nori_paths) / thread_num)
    npths_seperate = [nori_paths[i:i+inter] for i in range(0, len(nori_paths), inter)]
    del nori_paths

    for i in range(thread_num):
        t = SpeedupThread(npths_seperate[i])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default=None, help='bucket_name, startswith s3://.../')
    parser.add_argument('--pkl_path', default=None, help='pkl path, which saves nori paths')
    parser.add_argument('--include', help='*include*', nargs="*", default=[""])
    parser.add_argument('--exclude', help='*exclude*', nargs="*", default=None)
    parser.add_argument('-n', '--thread_num', type=int, default=4, help='the number of threads used for speeding up')
    args = parser.parse_args()

    speedup(args)


if __name__ == '__main__':
    # python3 nori_speedup.py --prefix s3://datamaid-tnt/datasets/action_s4 --include human
    main()
