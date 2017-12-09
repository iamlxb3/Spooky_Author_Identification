import os
import sys
import pickle
import torch
import random
import matplotlib.pyplot as plt

import io
import collections
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
language_dict_dir = os.path.join(top_dir, 'language_dict')

sys.path.append(top_dir)

# unigram
EAP_unigram_dict_path = os.path.join(language_dict_dir, 'train_unigram_dict_EAP')
HPL_unigram_dict_path = os.path.join(language_dict_dir, 'train_unigram_dict_HPL')
MWS_unigram_dict_path = os.path.join(language_dict_dir, 'train_unigram_dict_MWS')
#

# bigram
EAP_bigram_dict_path = os.path.join(language_dict_dir, 'train_bigram_dict_EAP')
HPL_bigram_dict_path = os.path.join(language_dict_dir, 'train_bigram_dict_HPL')
MWS_bigram_dict_path = os.path.join(language_dict_dir, 'train_bigram_dict_MWS')
#
