
import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import matplotlib.pyplot as plt
import traceback
import argparse
import os
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--phone", type=str, required=True)
  parser.add_argument("--mos", type=str, required=True)
  parser.add_argument("--spk", type=str, required=True)
  parser.add_argument("--output", type=str, required=True)
  args = parser.parse_args()

  ratios = []

  mos_file = open(args.mos, 'r').read().splitlines()
  mos = {}
  for line in mos_file:
    try:
      file_path, mos_score = line.split('|')
      filename = os.path.basename(file_path).split('.')[0]
      mos[filename] = float(mos_score)
    except:
      print(line)

  spk_file = open(args.spk).read().splitlines()
  spk = {}
  for line in spk_file:
    try:
      file_path, score = line.split('|')
      filename = os.path.basename(file_path).split('.')[0]
      spk[filename] = float(score)
    except:
      print(line)

  buffer = ""
  out_file = open(args.output, 'w')
  for line in tqdm(open(args.phone, errors='ignore').read().splitlines()):
    try:
      filepath, text, phone, language, confidence, ratio = line.split('|')
      confidence = float(confidence)
      ratio = float(ratio)
      filename = os.path.basename(filepath).split('.')[0]
      mos_score = mos[filename]
      spk_score = spk[filename]

      buffer += f"{filepath}|{text}|{phone}|{mos_score:.3f}|{language}|{confidence:.3f}|{spk_score:.3f}|{ratio:.3f}\n"
      if len(buffer) > 100000:
        out_file.write(buffer)
        buffer = ""
      ratios.append(ratio)
    except Exception as e:
      print(e, line)
      traceback.print_exc()
      continue
  out_file.write(buffer)
