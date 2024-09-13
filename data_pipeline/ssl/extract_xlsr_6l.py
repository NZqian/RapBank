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
from xlsr300m import WAV2VEC2_XLSR_300M
import sys

LOGGING_INTERVAL = 10
OFFSET = 0
BATCH_SIZE = 1


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
FEATURE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "xlsr_bgm_6l")
#KM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "xlsr_24l_512")
NUM_THREADS = int(sys.argv[3])

os.environ["OMP_NUM_THREADS"] = "4"

os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
#os.makedirs(KM_OUTPUT_DIR, exist_ok=True)

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

def inference(rank, queue: mp.Queue):

    def get_audio(path):
        wav, _ = librosa.load(path, sr=16000)

        wav = torch.FloatTensor(wav)
        return wav

    # device = torch.device("cuda", OFFSET + rank)
    #device = torch.device("cpu")
    device = torch.device(f"cuda:{rank}")

    # bundle=torchaudio.pipelines.WAV2VEC2_XLSR_300M
    bundle = WAV2VEC2_XLSR_300M
    bundle._normalize_waveform=False
    xlsr=bundle.get_model(dl_kwargs={'model_dir':'.','map_location':'cpu'})
    #xlsr=bundle.get_model(dl_kwargs={'model_dir':'/datablob/v-ziqianning/ckpts','map_location':'cpu'})
    xlsr = xlsr.eval()
    xlsr = xlsr.requires_grad_(False)
    xlsr = xlsr.to(device)


    while True:
        paths = queue.get()
        if paths is None:
            break

        #try:
        #    if os.path.exists(FEATURE_OUTPUT_DIR / f"{file_names[0]}.npy"):
        #        _ = np.load(FEATURE_OUTPUT_DIR / f"{file_names[0]}.npy")
        #        continue
        #except:
        #    pass

        try:
            file_names = [path.stem for path in paths]
            samples = [get_audio(path) for path in paths]
            lengths = [math.ceil(sample.shape[-1] / 320) for sample in samples]
            batched_samples = torch.nn.utils.rnn.pad_sequence(
                samples, batch_first=True
            ).to(device)


            features = xlsr.extract_features(batched_samples,lengths=None,num_layers=6)[0][-1]
            #features = xlsr.extract_features(batched_samples,lengths=None,num_layers=24)[0][-1]
            # [batch, frame, dim] of layer

            b, t, d = features.shape

            for feature, file_name, length in zip(
                features.cpu().numpy(), file_names, lengths
            ):                
                np.save(os.path.join(FEATURE_OUTPUT_DIR, f"{file_name}.npy"), feature) # [:length, :])
                #km_feat = apply_kmeans(feature)
                #np.save(os.path.join(KM_OUTPUT_DIR, f"{file_name}.npy"), km_feat) # [:length, :])

        except Exception as e:
            print(f"{e} in {paths} with longest length of {max(lengths)}")








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
    mp.set_start_method('spawn',force=True)

    gpu_num = torch.cuda.device_count()


    print(f"Running with {NUM_THREADS} threads and batchsize {BATCH_SIZE}")
    processes = []
    queue = mp.Queue()
    for thread_num in range(NUM_THREADS):

        rank = thread_num % gpu_num
        p = mp.Process(target=inference, args=(rank, queue))
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
        path_list = glob.glob(os.path.join(INPUT_DIR, '*.wav'))

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
    bar = tqdm(total=last_batches, desc="ssl")
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()

    #f_w = open(FILE_LIST,'a')
    #f_w.writelines(tmp_file)
    #f_w.close()












