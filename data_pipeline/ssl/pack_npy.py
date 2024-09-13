from npy_append_array import NpyAppendArray
import sys
import os
import numpy as np
from tqdm import tqdm
import random

in_dir = sys.argv[1]
out_file = sys.argv[2]
out_len_file = sys.argv[3]
percentage = float(sys.argv[4])


out_len_file = open(out_len_file, 'a')


#with NpyAppendArray(out_file, delete_if_exists=True) as npaa:
cnt = 0
with NpyAppendArray(out_file, delete_if_exists=True) as npaa:
    for file in tqdm(os.listdir(in_dir)):
        try:
        
            #if percentage > 0 and random.random() > percentage:
            #    continue
            arr = np.load(os.path.join(in_dir, file))
            if percentage > 0:
                indices = np.random.choice(arr.shape[0], int(arr.shape[0] * percentage), replace=False)
                arr = arr[indices]
            npaa.append(arr)
            out_len_file.write(f"{str(arr.shape[0])}\n")
            cnt += 1
        except:
            continue
    
data = np.load(out_file, mmap_mode="r")

print(data.shape)
print(len(os.listdir(in_dir)), cnt)
