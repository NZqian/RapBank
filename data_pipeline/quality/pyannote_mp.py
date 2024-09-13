import sys
import os
import torch
import torch.multiprocessing as mp
import multiprocessing
import threading
import numpy as np
import glob
import argparse
from tqdm import tqdm
from collections import defaultdict
import traceback
from pyannote.audio import Pipeline

file_lock = multiprocessing.Lock()


def inference(rank, text_path, queue: mp.Queue):
    device=f"cuda:{rank}"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="Your huggingface token")
    pipeline.to(torch.device(device))

    def write_to_file(data):
        with file_lock:
            with open(text_path, 'a') as f:
                f.write(data)

    buffer = ""
    
    with torch.no_grad():
        while True:
            #print(texts)
            filename = queue.get()
            if filename is None:
                write_to_file(buffer)
                break
            try:
                filename = filename[0]
                audio_path = filename

                spks = defaultdict(float)
                total_duration = 0.

                diarization = pipeline(audio_path)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    spks[speaker] += duration
                    total_duration += duration

                if len(spks) == 0:
                    percentage = 0.
                else:
                    sorted_spks = sorted(spks.items(), key=lambda s:s[1], reverse=True)
                    percentage = sorted_spks[0][1] / total_duration

                buffer += f"{filename}|{percentage:3}\n"
                if len(buffer) > 10000:
                    write_to_file(buffer)
                    buffer = ""
            except Exception as e:
                #print(sorted_spks)
                traceback.print_exc()


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

@setInterval(5)
def QueueWatcher(queue, bar):
    global last_batches
    curr_batches = queue.qsize()
    bar.update(last_batches-curr_batches)
    last_batches = curr_batches


if __name__ == "__main__":
    #audio_dir = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_or_dir", type=str, required=True)
    parser.add_argument("--text_path", type=str, required=True, help="Dir to save output")
    parser.add_argument("--jobs", type=int, required=False, default=2)
    parser.add_argument("--log_dir", type=str, required=False, help="For aml compatibility")
    parser.add_argument("--model_dir", type=str, required=False, help="For aml compatibility")
    args = parser.parse_args()

    mp.set_start_method('spawn',force=True)

    filelist_or_dir = args.filelist_or_dir
    text_path = args.text_path
    jobs = args.jobs
    os.makedirs(text_path, exist_ok=True)
    
    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = open(filelist_or_dir).read().splitlines()
        text_path = os.path.join(text_path, f"{filelist_name}_spk.txt")
    else:
        filelist_name = "single"
        generator = glob.glob(f"{filelist_or_dir}/*.wav")
        text_path = os.path.join(text_path, "spk.txt")

    os.system(f"rm {text_path}")

    gpu_num = torch.cuda.device_count()

    processes = []
    queue = mp.Queue()
    for thread_num in range(jobs):
        
        rank = thread_num % gpu_num
        p = mp.Process(target=inference, args=(rank, text_path, queue))
        p.start()
        processes.append(p)

    accum = []
    tmp_file = []

    for filename in generator:
        accum.append(filename)
        if len(accum) == 1:
            queue.put(accum.copy())
            accum.clear()


    for _ in range(jobs):
        queue.put(None)

    last_batches = queue.qsize()
    bar = tqdm(total=last_batches, desc='pyannote')
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()