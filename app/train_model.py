import os
import random
import sys
import numpy as np
import pickle
import torch
import random
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.neural_network import MLPClassifier

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
sys.path.append(top_dir)

from model.rnn_pytorch import RNNPytorch

data_dir = os.path.join(top_dir, 'data')
all_data_in_numpy_dir = os.path.join(data_dir, 'all_data_in_numpy')


# ----------------------------------------------------------------------------------------------------------------------
# split train, test
# ----------------------------------------------------------------------------------------------------------------------
sentence_id_list = os.listdir(all_data_in_numpy_dir)
train_sentence_id_list = sentence_id_list[0:140]
validation_sentence_id_list = sentence_id_list[14000:]
train_sentence_id_paths = [os.path.join(all_data_in_numpy_dir, x) for x in train_sentence_id_list]
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# build RNN
# ----------------------------------------------------------------------------------------------------------------------
n_hidden = 128
input_size = 22453
output_size = 3
rnn = RNNPytorch(input_size, n_hidden, output_size)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------------------------
MAX_ITER = 1000

for i in range(MAX_ITER):
    train_sample_path = random.sample(train_sentence_id_paths, 1)[0]
    f = pickle.load(open(train_sample_path, 'rb'))
    # output
    output, summed_word_one_hot_vector, words_one_hot_vector_in_sequence = f
    output = list(output).index(1) # convert np.array([1,0,0] to 1
    actual_output = torch.ByteTensor([output])
    # one-hot-vector-in-sequence
    input = torch.from_numpy(words_one_hot_vector_in_sequence)
    input = input.view(input.size()[0], 1, -1)
    print (actual_output)
    print (input)
    sys.exit()
# ----------------------------------------------------------------------------------------------------------------------





















