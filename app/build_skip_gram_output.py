import os
import sys
import pandas as pd
import nltk
import re
import collections
import pickle
import copy
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')



current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')
train_data_path = os.path.join(data_dir, 'train.csv')
train_word_dict_0_count_file_path = os.path.join(data_dir, 'train_word_dict_0_count')
train_word_dict_file_path = os.path.join(data_dir, 'train_word_dict')

skip_gram_output_dict_file_path = os.path.join(data_dir, 'skip_gram_output_dict')

df1 = pd.read_csv(train_data_path)
all_train_text = df1['text'].values

IS_LEMMA = True
lmtzr = WordNetLemmatizer()

# # ----------------------------------------------------------------------------------------------------------------------
# # build dictionary
# # ----------------------------------------------------------------------------------------------------------------------
# train_word_dict_0_count = collections.defaultdict(lambda :0)
# train_word_dict = collections.defaultdict(lambda :0)
#
# for text in all_train_text:
#     word_list = nltk.word_tokenize(text)
#     for word in word_list:
#         word = word.lower()
#         if IS_LEMMA:
#             word = lmtzr.lemmatize(word)
#         train_word_dict[word] += 1
#         train_word_dict_0_count[word] = 0
#
#
# pickle.dump(dict(train_word_dict_0_count), open(train_word_dict_0_count_file_path, 'wb'))
# pickle.dump(dict(train_word_dict), open(train_word_dict_file_path, 'wb'))
#
# # ----------------------------------------------------------------------------------------------------------------------

#print (train_word_dict.keys())


# ----------------------------------------------------------------------------------------------------------------------
# build skip-gram
# ----------------------------------------------------------------------------------------------------------------------
def _process_word(word):
    word = word.lower()
    if IS_LEMMA:
        word = lmtzr.lemmatize(word)
    return word


word_dict_0_count = pickle.load(open(train_word_dict_0_count_file_path, 'rb'))
word_dict_with_count = pickle.load(open(train_word_dict_file_path, 'rb'))


# add more keys
skip_gram_len = 3
START_LIST = ['start_{}'.format(x-skip_gram_len+1) for x in range(skip_gram_len)]
END_LIST = ['end_{}'.format(x) for x in range(skip_gram_len)]
for start_word in START_LIST:
    word_dict_0_count[start_word] = 0
for end_word in END_LIST:
    word_dict_0_count[end_word] = 0

# get word index
ordered_words = sorted(word_dict_0_count.keys())
word_total_count = len(ordered_words)
word_index_dict = {word:ordered_words.index(word) for word in ordered_words}

# build skip_gram_output_dict
skip_gram_output_dict = collections.defaultdict(lambda: np.zeros(word_total_count))


# get set
start_end_set = set(START_LIST+END_LIST)

all_train_text_len = len(all_train_text)
# count the surround words (length depends) for each word
for j, text in enumerate(all_train_text):
    word_list = nltk.word_tokenize(text)
    word_list = [_process_word(x) for x in word_list]
    word_list = START_LIST + word_list + END_LIST

    for i, word in enumerate(word_list):
        if word in start_end_set:
            continue
        else:
            surrounding_word_list = word_list[i-skip_gram_len:i] + word_list[i:i+skip_gram_len]
            for surrounding_word in surrounding_word_list:
                word_index = word_index_dict[surrounding_word]
                skip_gram_output_dict[word][word_index] += 1

    print ("text-{}/{} done!".format(j, all_train_text_len))



# convert values to numpy
for word, count_array in skip_gram_output_dict.items():
    word_count = int(word_dict_with_count[word])
    skip_gram_output_dict[word] = count_array / word_count
#

pickle.dump(dict(skip_gram_output_dict), open(skip_gram_output_dict_file_path, 'wb'))



print (skip_gram_output_dict['dirty'])
# ----------------------------------------------------------------------------------------------------------------------
