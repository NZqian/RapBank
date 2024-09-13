# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch

import torchaudio
import glob
import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
import joblib
import librosa
import threading
import math
import numpy as np
import itertools
from tqdm import tqdm
from pathlib import Path
import random
import os
import sys

LOGGING_INTERVAL = 10
OFFSET = 0
NUM_THREADS = 16
BATCH_SIZE = 1


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

INPUT_DIR = sys.argv[1]
FEATURE_OUTPUT_DIR = Path(sys.argv[2])

os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

def inference(rank, queue: mp.Queue):
    apply_kmeans = ApplyKmeans("km_xlsr_1024_18l")


    while True:
        paths = queue.get()
        if paths is None:
            break
        file_path = paths[0]
        file_name = os.path.basename(file_path)

        try:
            feat = np.load(file_path)
            km_feat = apply_kmeans(feat)
            np.save(FEATURE_OUTPUT_DIR / f"{file_name}", km_feat) # [:length, :])

        except Exception as e:
            print(e)
            raise
            #print(f"{e} in {paths} with longest length of {max(lengths)}")








def setInterval(interval):
    def decorator(function):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()

            def loop():  # executed in another thread
                while not stopped.wait(interval):  # until stopped
                    function(*args, **kwargs)

            t = threading.Thread(target=loop)
            t.daemon = True  # stop if the program exits
            t.start()
            return stopped

        return wrapper

    return decorator


last_batches = None


@setInterval(LOGGING_INTERVAL)
def QueueWatcher(queue):
    global last_batches
    curr_batches = queue.qsize()
    print(
        f"Remain: {curr_batches} batches [ {(last_batches-curr_batches)/LOGGING_INTERVAL} batches/s ]"
    )
    last_batches = curr_batches


if __name__ == "__main__":
    mp.set_start_method('spawn',force=True)
    FEATURE_OUTPUT_DIR.mkdir(exist_ok=True)

    gpu_num = torch.cuda.device_count()


    print(f"Running with {NUM_THREADS} threads and batchsize {BATCH_SIZE}")
    processes = []
    queue = mp.Queue()
    for thread_num in range(NUM_THREADS):

        #rank = thread_num % gpu_num
        p = mp.Process(target=inference, args=(thread_num, queue))
        p.start()
        processes.append(p)

    accum = []
    tmp_file = []
    
    # path_list = []
    # for scp in glob.glob(os.path.join(INPUT_DIR, '*.lst')):
    #     tmp = [x.split('\t')[0] for x in open(scp).readlines()]
    #     print(len(tmp))
    #     path_list = list(set(path_list) | set(tmp))
    #     print(len(path_list))



    if os.path.isfile(INPUT_DIR):
        path_list = [x.strip() for x in open(INPUT_DIR).readlines()]
    else:
        #path_list = glob.glob(os.path.join(INPUT_DIR, '*.wav'))
        path_list = [os.path.join(INPUT_DIR, x) for x in os.listdir(INPUT_DIR)]

    # for file in tqdm(INPUT_DIR.glob("**/*.wav")):
    for file in tqdm(path_list):
        file = Path(file)
        # if not input_guard(file):
        #     continue
        accum.append(file)
        if len(accum) == BATCH_SIZE:
            queue.put(accum.copy())
            accum.clear()
        # tmp_file.append(file.as_posix()+'\n')

    for _ in range(NUM_THREADS):
        queue.put(None)

    last_batches = queue.qsize()
    queue_watcher = QueueWatcher(queue)
    for p in processes:
        p.join()
    queue_watcher.set()

    #f_w = open(FILE_LIST,'a')
    #f_w.writelines(tmp_file)
    #f_w.close()












