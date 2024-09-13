import sys
import os
import torch
import torch.multiprocessing as mp
import threading
import numpy as np
import glob
import argparse
import librosa
import soxr
from tqdm import tqdm
import traceback
import multiprocessing
#from speechmos import dnsmos
import onnxruntime as ort
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

file_lock = multiprocessing.Lock()

SR = 16000
INPUT_LENGTH = 9.01
dnsmos = None


class DNSMOS:
    def __init__(self, primary_model_path, p808_model_path, rank) -> None:
        self.primary_model_path = primary_model_path
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        #providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),})]
        #providers = ["CUDAExecutionProvider"]
        #providers = ["CPUExecutionProvider"]
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': rank,
            }),
            'CPUExecutionProvider',
        ]
        #self.onnx_sess = ort.InferenceSession(self.primary_model_path, sess_opt, providers=providers)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, sess_opt, providers=providers)
        #print(self.p808_onnx_sess.get_providers())

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d(
                [-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d(
                [-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d(
                [-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, sample, fs, is_personalized_MOS):
        clip_dict = {}
        if isinstance(sample, np.ndarray):
            audio = sample
            if not ((audio >= -1).all() and (audio <= 1).all()):
                raise ValueError("np.ndarray values must be between -1 and 1.")
        elif isinstance(sample, str) and os.path.isfile(sample):
            audio, _ = librosa.load(sample, sr=fs)
            clip_dict['filename'] = sample
        else:
            raise ValueError(
                f"Input must be a numpy array or a path to an audio file.")

        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype(
                'float32')[np.newaxis, :]
            p808_input_features = np.array(self.audio_melspec(
                audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            #mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[
            #    0][0]
            #mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
            #    mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
            #predicted_mos_sig_seg.append(mos_sig)
            #predicted_mos_bak_seg.append(mos_bak)
            #predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        #clip_dict['ovrl_mos'] = np.mean(predicted_mos_ovr_seg)
        #clip_dict['sig_mos'] = np.mean(predicted_mos_sig_seg)
        #clip_dict['bak_mos'] = np.mean(predicted_mos_bak_seg)
        clip_dict['p808_mos'] = np.mean(predicted_p808_mos)
        return clip_dict

def normalize_audio(y, target_dbfs=0):
    max_amplitude = np.max(np.abs(y))
    if max_amplitude < 0.1:
        return y

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude
    #print(max_amplitude, target_amplitude, scale_factor)

    normalized_audio = y * scale_factor

    return normalized_audio


def inference(rank, ckpt_dir, text_path, queue: mp.Queue):
    p808_model_path = os.path.join(ckpt_dir, 'dnsmos_p808.onnx')
    primary_model_path = os.path.join(ckpt_dir, 'sig_bak_ovr.onnx')
    dnsmos = DNSMOS(primary_model_path, p808_model_path, rank)

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
                wav, sr = librosa.load(audio_path, sr=None)
                wav = normalize_audio(wav, -6)
                wav = soxr.resample(
                    wav,          # 1D(mono) or 2D(frames, channels) array input
                    sr,      # input samplerate
                    16000       # target samplerate
                )
                if wav.min() < -1 or wav.min() > 1:
                    print(audio_path)
                mos_dict = dnsmos(wav, 16000, False)
                p808_mos = mos_dict['p808_mos']
                buffer += f"{filename}|{p808_mos:3}\n"
                if len(buffer) > 10000:
                    write_to_file(buffer)
                    buffer = ""
            except Exception as e:
                print(audio_path)
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
    parser.add_argument("--ckpt_path", type=str, required=False, default=".")
    args = parser.parse_args()

    mp.set_start_method('spawn',force=True)

    filelist_or_dir = args.filelist_or_dir
    text_path = args.text_path
    jobs = args.jobs
    ckpt_path = args.ckpt_path
    os.makedirs(text_path, exist_ok=True)
    
    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = open(filelist_or_dir).read().splitlines()
        text_path = os.path.join(text_path, f"{filelist_name}_dnsmos.txt")
    else:
        filelist_name = "single"
        generator = glob.glob(f"{filelist_or_dir}/*.wav")
        text_path = os.path.join(text_path, "dnsmos.txt")

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
    bar = tqdm(total=last_batches, desc='dnsmos')
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()