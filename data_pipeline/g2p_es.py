import re
from unidecode import unidecode
from transformers import T5ForConditionalGeneration, AutoTokenizer
import matplotlib.pyplot as plt
import traceback
import sys
import os
from tqdm import tqdm
import numpy as np


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)

puncs_to_remove = ["♪", "#", "¿", "¡", "-", "*"]
puncs_to_remove = "".join(puncs_to_remove)
def normalize(text):
  text = text.translate(str.maketrans('', '', puncs_to_remove))
  text = text.strip()
  return text


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''


if __name__ == '__main__':
  text_file = sys.argv[1]
  phoneme_file = sys.argv[2]


  model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
  #model.cuda()
  tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

  buffer = ""

  out_file = open(phoneme_file, 'w')
  for line in tqdm(open(text_file, errors='ignore').read().splitlines()):
    try:
      filepath, text, language, confidence = line.split('|')
      confidence = float(confidence)
      filename = os.path.basename(filepath).split('.')[0]
      duration = float(filename.split('_')[-1]) / 1000

      if language == "es":
        #text = convert_to_ascii(text)
        text = normalize(text)
        text = lowercase(text)
        print(text)

        words = text.split(' ')
        words = ['<spa>: '+i for i in words]
        out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')

        preds = model.generate(**out,num_beams=1,max_length=50) # We do not find beam search helpful. Greedy decoding is enough. 
        phone = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)
        phone = " ".join(phone)
        print(phone)

        phone = collapse_whitespace(phone)
        ratio = len(phone) / duration
      else:
        phone = "[blank]"
        ratio = 0
      buffer += f"{filepath}|{text}|{phone}|{language}|{confidence:.3f}|{ratio:.3f}\n"
      if len(buffer) > 100000:
        out_file.write(buffer)
        buffer = ""
      #break
    except Exception as e:
      print(filename, line, e)
      continue
  out_file.write(buffer)