import webrtcvad
import torch.multiprocessing as mp
import os
import threading
from tqdm import tqdm
import sys
from scipy.io.wavfile import write
import traceback
import librosa
import argparse
import glob
import time
import random

vocal_file_lock = threading.Lock()
bgm_file_lock = threading.Lock()

from vad_tool import read_wave_to_frames, read_wave_to_frames_withbgm, vad_generator, cut_points_generator, cut_points_storage_generator, wavs_generator

LOGGING_INTERVAL = 3
#SAMPLE_RATE = 44100
#SAMPLE_RATE = 16000
SAMPLE_RATE = 48000
SAVE_SAMPLE_RATE = 44100
FRAME_DURATION = 10

SAVE_SAMPLE_PER_FRAME = int(FRAME_DURATION * SAVE_SAMPLE_RATE / 1000)

MIN_ACTIVE_TIME_MS = 200
SIL_HEAD_TAIL_MS = 500
#SIL_HEAD_TAIL_MS = 3000
SIL_MID_MS = 3000
CUT_MIN_MS = 3000
CUT_MAX_MS = 30000

MIN_ACTIVE_FRAME = MIN_ACTIVE_TIME_MS // FRAME_DURATION
SIL_FRAME = SIL_HEAD_TAIL_MS // FRAME_DURATION
SIL_MID_FRAME = SIL_MID_MS // FRAME_DURATION
CUT_MIN_FRAME = CUT_MIN_MS // FRAME_DURATION
CUT_MAX_FRAME = CUT_MAX_MS // FRAME_DURATION
RANDOM_MIN_FRAME = True

import torch

def gpu_holder(rank, a):
    device=f'cuda:{rank}'
    conv = torch.nn.Conv1d(1024, 1024, 9, padding=4)
    conv.to(device)
    while True:
        x = torch.rand((8, 1024, 128), device=device)
        y = conv(x)



    
def inference(rank, out_dir, filelist_name, queue: mp.Queue):
    vocal_out_dir = os.path.join(out_dir, "vocal_cut")
    bgm_out_dir = os.path.join(out_dir, "bgm_cut")
    info_dir = os.path.join(out_dir, "vad_info")
    os.makedirs(vocal_out_dir, exist_ok=True)
    os.makedirs(bgm_out_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    def write_to_file(file_path, data, file_lock):
        with file_lock:
            with open(file_path, 'a') as f:
                f.write(data)
    while True:
        input_path = queue.get()
        if input_path is None:
            break
        try:
            vad_tools = webrtcvad.Vad(3) # create a new vad each time to avoid some bugs
            vocal_path, bgm_path = input_path[0]
            filename = os.path.basename(vocal_path).replace(".wav", "")
            #frames, wav = read_wave_to_frames(vocal_path, SAMPLE_RATE, FRAME_DURATION)
            frames, wav, vocal_wav, bgm_wav = read_wave_to_frames_withbgm(vocal_path, bgm_path, SAMPLE_RATE, SAVE_SAMPLE_RATE, FRAME_DURATION)
            vad_info = vad_generator(frames, SAMPLE_RATE, vad_tools)

            cut_points = cut_points_generator(vad_info, MIN_ACTIVE_FRAME, SIL_FRAME, SIL_MID_FRAME, CUT_MIN_FRAME, CUT_MAX_FRAME, RANDOM_MIN_FRAME)
            raw_vad_content, file_content = cut_points_storage_generator(vad_info, cut_points, FRAME_DURATION)

            with open(os.path.join(info_dir, filename+".raw_info.txt"), "w") as f:
                f.write(raw_vad_content)
            with open(os.path.join(info_dir, filename+".txt"), "w") as f:
                f.write(file_content)

            wavs = wavs_generator(vocal_wav, cut_points, filename, SAVE_SAMPLE_RATE, FRAME_DURATION)
            bgm_wavs = wavs_generator(bgm_wav, cut_points, filename, SAVE_SAMPLE_RATE, FRAME_DURATION)
            for ((wav_seg, name), (bgm_wav_seg, _)) in zip(wavs, bgm_wavs):
                if wav_seg.shape[-1] < SAVE_SAMPLE_RATE * CUT_MIN_MS / 1000:
                    continue
                write(os.path.join(vocal_out_dir, name), SAVE_SAMPLE_RATE, wav_seg)
                write(os.path.join(bgm_out_dir, name), SAVE_SAMPLE_RATE, bgm_wav_seg)

        except Exception as e:
            traceback.print_exc()
            print(e)

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
def QueueWatcher(queue, bar):
    global last_batches
    curr_batches = queue.qsize()
    bar.update(last_batches-curr_batches)
    last_batches = curr_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_or_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--jobs", type=int, required=False, default=2, help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    parser.add_argument("--model_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    args = parser.parse_args()

    filelist_or_dir = args.filelist_or_dir
    out_dir = args.out_dir
    NUM_THREADS = args.jobs

    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = [os.path.basename(x) for x in open(filelist_or_dir).read().splitlines()]
    else:
        filelist_name = "single"
        generator = [(os.path.join(os.path.dirname(os.path.dirname(x)), "vocal", os.path.basename(x)), os.path.join(os.path.dirname(os.path.dirname(x)), "bgm", os.path.basename(x))) for x in glob.glob(f"{filelist_or_dir}/*.wav")]
    
    #mp.set_start_method('spawn',force=True)

    print(f"Running with {NUM_THREADS} threads and batchsize 1")
    processes = []
    queue = mp.Queue()
    for rank in range(NUM_THREADS):
        p = mp.Process(target=inference, args=(rank, out_dir, filelist_name, queue), daemon=True)
        p.start()
        processes.append(p)

    for i in range(4):
        rank = i % torch.cuda.device_count()
        p = mp.Process(target=gpu_holder, args=(rank, 0), daemon=True)
        p.start()
        #processes.append(p)

    accum = []
    tmp_file = []
    

    for filename in tqdm(generator):
        #accum.append((os.path.join(out_dir, "vocal", filename), os.path.join(out_dir, "bgm", filename)))
        accum.append(filename)
        if len(accum) == 1:
            queue.put(accum.copy())
            accum.clear()


    for _ in range(NUM_THREADS):
        queue.put(None)

    last_batches = queue.qsize()
    bar = tqdm(total=last_batches)
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()
