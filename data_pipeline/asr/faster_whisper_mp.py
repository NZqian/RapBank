import sys
import os
#from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import threading
#import librosa
#import numpy as np
from faster_whisper import WhisperModel
#import whisper
import glob
import fcntl
import argparse
import traceback
from tqdm import tqdm
import numpy as np
import librosa
import soxr
import multiprocessing

def normalize_audio(y, target_dbfs=0):
    max_amplitude = np.max(np.abs(y))
    if max_amplitude < 0.1:
        return y

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio
file_lock = multiprocessing.Lock()

def inference(rank, ckpt_path, text_path, queue: mp.Queue):
    device = f"cuda"
    model = WhisperModel(ckpt_path, device=device, device_index=rank, compute_type="float16")
    puncs = list(",.?!")
    buffer = ""
    def write_to_file(data):
        with file_lock:
            with open(text_path, 'a') as f:
                f.write(data)

    
    with torch.no_grad():
        while True:
            #print(texts)
            filename = queue.get()
            if filename is None:
                write_to_file(buffer)
                break
            filename = filename[0]

            try:
                audio_path = filename
                audio, sr = librosa.load(audio_path, sr=None)
                audio = normalize_audio(audio, -6)
                audio = soxr.resample(
                    audio,
                    sr,
                    16000
                )
                segments, info = model.transcribe(audio, beam_size=3, vad_filter=True, condition_on_previous_text=False)
                text = ""

                for segment in segments:
                    text_segment = segment.text
                    text_segment.strip()
                    if len(text_segment) == 0:
                        continue
                    if not text_segment[-1] in puncs:
                        text_segment += ","
                    text = text + " " + text_segment
                text = text.replace("  ", " ")
                text = text.strip()
                if len(text) == 0:
                    continue
                if text[-1] == ",":
                    text = text[:-1] + "."

                buffer += f"{filename}|{text}|{info.language}|{info.language_probability}\n"
                if len(buffer) > 10000:
                    write_to_file(buffer)
                    buffer = ""
            
            except Exception as e:
                print(filename)
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
    parser.add_argument("--jobs", type=int, required=False, default=2, help="Path to save checkpoints")
    parser.add_argument("--ckpt_path", type=str, required=False, default="large-v3")
    parser.add_argument("--log_dir", type=str, required=False, default="large-v3", help="For aml compability")
    parser.add_argument("--model_dir", type=str, required=False, default="large-v3", help="For aml compability")
    args = parser.parse_args()

    mp.set_start_method('spawn',force=True)

    filelist_or_dir = args.filelist_or_dir
    text_path = args.text_path
    jobs = args.jobs
    ckpt_path = args.ckpt_path
    os.makedirs(text_path, exist_ok=True)
    model = WhisperModel(ckpt_path, device='cpu') # download model in one thread
    del(model)
    
    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = open(filelist_or_dir).read().splitlines()
        text_path = os.path.join(text_path, f"{filelist_name}_text.txt")
    else:
        filelist_name = "single"
        generator = glob.glob(f"{filelist_or_dir}/*.wav")
        text_path = os.path.join(text_path, "text.txt")

    os.system(f"rm {text_path}")

    gpu_num = torch.cuda.device_count()

    processes = []
    queue = mp.Queue()
    for thread_num in range(jobs):
        
        rank = thread_num % gpu_num
        p = mp.Process(target=inference, args=(rank, ckpt_path, text_path, queue))
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
    bar = tqdm(total=last_batches, desc='whisper')
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()