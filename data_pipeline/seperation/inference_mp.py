import torch
import torch.multiprocessing as mp
import os, sys
import threading
from tqdm import tqdm
import soundfile as sf
import threading
import librosa
import numpy as np
from utils import demix_track, demix_track_demucs, get_model_from_config
import traceback
import glob
import argparse

import warnings
warnings.filterwarnings("ignore")

def normalize_audio(y, target_dbfs=0):
    max_amplitude = np.max(np.abs(y))
    if max_amplitude < 0.1:
        return y

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio

def inference(rank, ckpt_root, out_dir, queue: mp.Queue):
    #print(f"thread {rank} start")
    device = f"cuda:{rank}"
    config = f"{ckpt_root}/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    ckpt = f"{ckpt_root}/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    model, config = get_model_from_config("bs_roformer", config)
    state_dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    
    with torch.no_grad():
        while True:
            #print(texts)
            filename = queue.get()
            if filename is None:
                break
            filepath = filename[0]
            filename = filepath.split('/')[-1]
            try:
                mix, sr = librosa.load(filepath, sr=44100, mono=False)
                #mix = normalize_audio(mix, -6)
                mix = mix.T
                if len(mix.shape) == 1:
                    mix = np.stack([mix, mix], axis=-1)

                mixture = torch.tensor(mix.T, dtype=torch.float32)
                res = demix_track(config, model, mixture, device)
                sf.write("{}/{}".format(os.path.join(out_dir, "vocal"), filename), res['vocals'].T.mean(-1), sr, subtype='FLOAT')
                sf.write("{}/{}".format(os.path.join(out_dir, "bgm"), filename), mix.mean(-1) - res['vocals'].T.mean(-1), sr, subtype='FLOAT')

                
            except Exception as e:
                traceback.print_exc()
                continue



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

@setInterval(3)
def QueueWatcher(queue, bar):
    global last_batches
    curr_batches = queue.qsize()
    bar.update(last_batches-curr_batches)
    last_batches = curr_batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_or_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--jobs", type=int, required=False, default=2, help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    parser.add_argument("--model_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    args = parser.parse_args()

    filelist_or_dir = args.filelist_or_dir
    out_dir = args.out_dir
    ckpt_path = args.ckpt_path
    jobs = args.jobs
    vad_jobs = jobs * 2

    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = open(filelist_or_dir).read().splitlines()
    else:
        filelist_name = "single"
        generator = glob.glob(f"{filelist_or_dir}/*.wav")

    os.makedirs(os.path.join(out_dir, "vocal"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bgm"), exist_ok=True)


    gpu_num = torch.cuda.device_count()

    processes = []
    vad_processes = []
    queue = mp.Queue()
    vad_queue = mp.Queue()
    for thread_num in range(jobs):
        rank = thread_num % gpu_num
        p = mp.Process(target=inference, args=(rank, ckpt_path, out_dir, queue))
        p.start()
        processes.append(p)

    accum = []

    for filename in tqdm(generator):
        accum.append(filename)
        if len(accum) == 1:
            queue.put(accum.copy())
            accum.clear()

    for _ in range(jobs):
        queue.put(None)

    last_batches = queue.qsize()
    bar = tqdm(total=last_batches, desc="seperation")
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()

    for p in vad_processes:
        p.join()
