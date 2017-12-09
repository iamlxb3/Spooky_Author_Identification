import os
import sys
import pandas as pd
import pickle
import torch
import random
import math
import matplotlib.pyplot as plt

import io
import collections
from sklearn.metrics import accuracy_score

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')
language_dict_dir = os.path.join(data_dir, 'language_dict')

sys.path.append(top_dir)
from other_funcs.nltk_funcs import lemmatize_word, tokenize_word, generate_bigrams

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

# read validation csv
validation_data_path = os.path.join(data_dir, 'train_2.csv')
validation_df = pd.read_csv(validation_data_path)

# ----------------------------------------------------------------------------------------------------------------------
# initialize
# ----------------------------------------------------------------------------------------------------------------------

# unigram
EAP_unigram_dict = pickle.load(open(EAP_unigram_dict_path, 'rb'))
HPL_unigram_dict = pickle.load(open(HPL_unigram_dict_path, 'rb'))
MWS_unigram_dict = pickle.load(open(MWS_unigram_dict_path, 'rb'))
EAP_word_total = sum(EAP_unigram_dict.values())
HPL_word_total = sum(HPL_unigram_dict.values())
MWS_word_total = sum(MWS_unigram_dict.values())
EAP_V = len(EAP_unigram_dict.keys())
HPL_V = len(HPL_unigram_dict.keys())
MWS_V = len(MWS_unigram_dict.keys())
#

# bigram
EAP_bigram_dict = pickle.load(open(EAP_bigram_dict_path, 'rb'))
HPL_bigram_dict = pickle.load(open(HPL_bigram_dict_path, 'rb'))
MWS_bigram_dict = pickle.load(open(MWS_bigram_dict_path, 'rb'))
EAP_bigram_total = sum(EAP_bigram_dict.values())
HPL_bigram_total = sum(HPL_bigram_dict.values())
MWS_bigram_total = sum(MWS_bigram_dict.values())
# EAP_bigram_V = len(EAP_bigram_dict.keys())
# HPL_bigram_V = len(HPL_bigram_dict.keys())
# MWS_bigram_V = len(MWS_bigram_dict.keys())
#

AUTHOR_LIST = ['EAP', 'HPL', 'MWS']

def _calculate_unigram_word_score(word):

    # EAP
    if word in EAP_unigram_dict:
        word_count = EAP_unigram_dict[word]
        EAP_likelihood = word_count/EAP_word_total
    else:
        word_count = 1
        EAP_likelihood = word_count/(EAP_word_total + EAP_V)
    #

    # HPL
    if word in HPL_unigram_dict:
        word_count = HPL_unigram_dict[word]
        HPL_likelihood = word_count/HPL_word_total
    else:
        word_count = 1
        HPL_likelihood = word_count/(HPL_word_total + HPL_V)
    #

    # MWS
    if word in MWS_unigram_dict:
        word_count = MWS_unigram_dict[word]
        MWS_likelihood = word_count/MWS_word_total
    else:
        word_count = 1
        MWS_likelihood = word_count/(MWS_word_total + MWS_V)
    #
    return EAP_likelihood, HPL_likelihood, MWS_likelihood

def _calculate_bigram_word_score(bigram):
    # (word1, word2),    P(word2 | word1) = P(word1, word2) / P(word1)
    word1, word2 = bigram
    EAP_P_w1, HPL_P_w1, MWS_P_w1 = _calculate_unigram_word_score(word1)

    # EAP
    if bigram in EAP_bigram_dict:
        bigram_count = EAP_bigram_dict[bigram]
        P_w1_w2 = bigram_count/EAP_bigram_total
        EAP_likelihood = P_w1_w2/EAP_P_w1
    else:
        EAP_likelihood = None
    #

    # HPL
    if bigram in HPL_bigram_dict:
        bigram_count = HPL_bigram_dict[bigram]
        P_w1_w2 = bigram_count/HPL_bigram_total
        HPL_likelihood = P_w1_w2/HPL_P_w1
    else:
        HPL_likelihood = None
    #

    # MWS
    if bigram in MWS_bigram_dict:
        bigram_count = MWS_bigram_dict[bigram]
        P_w1_w2 = bigram_count/MWS_bigram_total
        MWS_likelihood = P_w1_w2/MWS_P_w1
    else:
        MWS_likelihood = None
    #

    return EAP_likelihood, HPL_likelihood, MWS_likelihood


def _process_word(word):
    word = word.lower()
    word = lemmatize_word(word)
    return word
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# unigram model
# ----------------------------------------------------------------------------------------------------------------------
actual_author_list = []
pred_author_list = []


for i, row in validation_df.iterrows():
    text = row['text']
    author = row['author']
    actual_author_list.append(author)
    word_list = tokenize_word(text)

    EAP_sentence_likelihood = 0.0
    HPL_sentence_likelihood = 0.0
    MWS_sentence_likelihood = 0.0

    for j, word in enumerate(word_list):
        word = _process_word(word)
        EAP_likelihood, HPL_likelihood, MWS_likelihood = _calculate_unigram_word_score(word)
        EAP_sentence_likelihood += math.log(EAP_likelihood)
        HPL_sentence_likelihood += math.log(HPL_likelihood)
        MWS_sentence_likelihood += math.log(MWS_likelihood)

    likelihood_list = [EAP_sentence_likelihood, HPL_sentence_likelihood, MWS_sentence_likelihood]
    max_index = likelihood_list.index(max(likelihood_list))
    pred_author = AUTHOR_LIST[max_index]
    pred_author_list.append(pred_author)

accuracy = accuracy_score(actual_author_list, pred_author_list)
print (collections.Counter(pred_author_list))
print ("unigram accuracy: ", accuracy)
# ----------a-----------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# stupid back-off bigram model
# ----------------------------------------------------------------------------------------------------------------------
actual_author_list = []
pred_author_list = []


for _, row in validation_df.iterrows():
    text = row['text']
    author = row['author']
    actual_author_list.append(author)
    word_list = tokenize_word(text)

    EAP_sentence_likelihood = 0.0
    HPL_sentence_likelihood = 0.0
    MWS_sentence_likelihood = 0.0

    for j, word in enumerate(word_list):
        word = _process_word(word)
        word_list[j] = word

    bigrams_list = generate_bigrams(word_list)


    for bigram in bigrams_list:
        likelihood_tuple = _calculate_bigram_word_score(bigram)
        likelihood_list = list(likelihood_tuple)

        # stupid back-off
        for index, likelihood in enumerate(likelihood_list):
            if likelihood is None:
                unigram_likelihood_tuple = _calculate_unigram_word_score(bigram[1])
                likelihood_list[index] = 0.4 * unigram_likelihood_tuple[index]

        EAP_likelihood, HPL_likelihood, MWS_likelihood = likelihood_list


        EAP_sentence_likelihood += math.log(EAP_likelihood)
        HPL_sentence_likelihood += math.log(HPL_likelihood)
        MWS_sentence_likelihood += math.log(MWS_likelihood)

    likelihood_list = [EAP_sentence_likelihood, HPL_sentence_likelihood, MWS_sentence_likelihood]
    max_index = likelihood_list.index(max(likelihood_list))
    pred_author = AUTHOR_LIST[max_index]
    pred_author_list.append(pred_author)

accuracy = accuracy_score(actual_author_list, pred_author_list)
print (collections.Counter(pred_author_list))
print ("back-off bigram accuracy: ", accuracy)






# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Kneser-Key smoothing model
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------