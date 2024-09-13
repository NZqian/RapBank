""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
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

  backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True)

  buffer = ""

  out_file = open(phoneme_file, 'w')
  for line in tqdm(open(text_file, errors='ignore').read().splitlines()):
    try:
      filepath, text, language, confidence = line.split('|')
      confidence = float(confidence)
      filename = os.path.basename(filepath).split('.')[0]
      duration = float(filename.split('_')[-1]) / 1000

      if language == "en":
        phone = convert_to_ascii(text)
        phone = lowercase(phone)
        phone = expand_abbreviations(phone)

        phone = backend.phonemize([phone], strip=True)[0]
        phone = collapse_whitespace(phone)
        ratio = len(phone) / duration
      else:
        phone = "[blank]"
        ratio = 0
      buffer += f"{filepath}|{text}|{phone}|{language}|{confidence:.3f}|{ratio:.3f}\n"
      if len(buffer) > 100000:
        out_file.write(buffer)
        buffer = ""
    except Exception as e:
      print(filename, line, e)
      continue
  out_file.write(buffer)