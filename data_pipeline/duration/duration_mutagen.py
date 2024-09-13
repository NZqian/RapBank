import os
import glob
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from mutagen.wave import WAVE # mutagen for reading wav metadata

filelist_or_dir = sys.argv[1] # filelist including absolute path or data root path

total_duration = 0.
durations = []


def get_wav_duration(file_path):
    try:
        duration = WAVE(file_path).info.length
        return duration
    except Exception as e:
        print('Error occurred:', e)
        return None

if os.path.isdir(filelist_or_dir):
    filelist = [os.path.join(filelist_or_dir, filename) for filename in glob.glob(os.path.join(filelist_or_dir, '**/*.wav'), recursive=True)]
else:
    filelist = open(filelist_or_dir, 'r').read().splitlines()
for wav_path in tqdm(filelist):
    try:
        duration = get_wav_duration(wav_path)
        total_duration += duration
        durations.append(duration)
    except Exception as e:
        print(e)

print(f"total_duration: {total_duration}, avg_duration: {total_duration / len(durations)}")

#plt.hist(durations, bins=50, range=(0, 50))
#plt.savefig(os.path.join(os.path.dirname(data_root), "durations.png"))
#np.save(os.path.join(os.path.dirname(data_root), "1.npy"), np.array(durations))