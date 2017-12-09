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

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')



current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')
train_data_path = os.path.join(data_dir, 'train_1.csv')
validation_data_path = os.path.join(data_dir, 'train_2.csv')

train_df = pd.read_csv(train_data_path)
validation_df = pd.read_csv(validation_data_path)





lmtzr = WordNetLemmatizer()
def _process_word(word):
    IS_LEMMA = True
    word = word.lower()
    if IS_LEMMA:
        word = lmtzr.lemmatize(word)
    return word



# get the train unigram and bigram dict

# unigram
all_train_words_dict = collections.defaultdict(lambda :[])
for index, row in train_df.iterrows():
    text = row['text']
    author = row['author']
    word_list = nltk.word_tokenize(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = lmtzr.lemmatize(word)
    all_train_words_dict[author].extend(word_list)

for author, all_train_words in all_train_words_dict.items():
    train_unigram_dict = collections.Counter(all_train_words)
    pickle.dump(train_unigram_dict, open(os.path.join(data_dir, 'language_dict', 'train_unigram_dict_{}'.
                                                      format(author)), 'wb'))
    print ('Save author-{} unigram done!'.format(author))
#

# bigram
all_train_phrases_dict = collections.defaultdict(lambda :[])
for index, row in train_df.iterrows():
    text = row['text']
    author = row['author']
    word_list = nltk.word_tokenize(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = lmtzr.lemmatize(word)
    word_list = ['start_0'] + word_list + ['end_0']
    phrases_list = []
    for i, word in enumerate(word_list):
        if i <= len(word_list) - 2:
            phrases = (word, word_list[i+1])
            phrases_list.append(phrases)

    all_train_phrases_dict[author].extend(phrases_list)

for author, all_train_phrases in all_train_phrases_dict.items():
    all_train_phrases_dict = collections.Counter(all_train_phrases)
    print (sorted(all_train_phrases_dict.items(), key=lambda x:x[1]))
    pickle.dump(all_train_phrases_dict, open(os.path.join(data_dir, 'language_dict', 'train_bigram_dict_{}'.format(author)), 'wb'))
    print ('Save author-{} bigram done!'.format(author))
#