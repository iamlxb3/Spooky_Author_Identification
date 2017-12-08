import os
import sys
import numpy as np
import pickle
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.neural_network import MLPClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')

skip_gram_output_dict_path = os.path.join(data_dir, 'skip_gram_output_dict')
train_word_dict_0_count_path = os.path.join(data_dir, 'train_word_dict_0_count')
word_index_dict_path = os.path.join(data_dir, 'word_index_dict')
train_data_path = os.path.join(data_dir, 'train.csv')
all_data_in_numpy_dir = os.path.join(data_dir, 'all_data_in_numpy')

word_index_dict = pickle.load(open(word_index_dict_path, 'rb'))

train_word_dict_0_count = pickle.load(open(train_word_dict_0_count_path, 'rb'))
word_total_num = len(train_word_dict_0_count.keys())

skip_gram_output_dict = pickle.load(open(skip_gram_output_dict_path, 'rb'))

for _, value in skip_gram_output_dict.items():
    skip_gram_output_total_num = len(value)
    break


IS_LEMMA = True
lmtzr = WordNetLemmatizer()


def _process_word(word):
    word = word.lower()
    if IS_LEMMA:
        word = lmtzr.lemmatize(word)
    return word

# author list
AUTHORS = ['EAP', 'HPL', 'MWS']


# get train data
train_df = pd.read_csv(train_data_path)

all_data_list = []
for i, (index, row) in enumerate(train_df.iterrows()):
    id = row['id']
    text = row['text']
    author = row['author']

    # map author to vector
    output = np.zeros(3, dtype=np.uint8)
    author_index = AUTHORS.index(author)
    output[author_index] = 1.0
    #

    #
    word_list = nltk.word_tokenize(text)
    word_list = [_process_word(x) for x in word_list]

    summed_words_vector = np.zeros(skip_gram_output_total_num)
    summed_word_one_hot_vector = np.zeros(word_total_num)

    word_vector_list = []
    word_one_hot_vector_list = []

    for word in word_list:
        # summed word vector
        word_vector = skip_gram_output_dict[word]
        summed_words_vector += word_vector
        # sequence word vector
        word_vector_list.append(word_vector)
        # one-hot word vector
        one_hot_vector = np.zeros(word_total_num, dtype=np.uint8)
        one_hot_word_index = word_index_dict[word]
        one_hot_vector[one_hot_word_index] = 1
        summed_word_one_hot_vector += one_hot_vector
        # one hot sequence
        word_one_hot_vector_list.append(one_hot_vector)


    words_vector_in_sequence = np.array(word_vector_list)
    words_one_hot_vector_in_sequence = np.array(word_one_hot_vector_list, dtype=np.uint8)

    # print (word_list)
    # print (output)
    # print(len(summed_word_one_hot_vector[summed_word_one_hot_vector > 0]))
    # print(words_one_hot_vector_in_sequence.shape)
    # print (len(summed_words_vector[summed_words_vector>0]))
    # print (words_vector_in_sequence.shape)

    #data_tuple = (id, word_list, output, summed_word_one_hot_vector, words_one_hot_vector_in_sequence,
    #                      summed_words_vector, words_vector_in_sequence)
    data_tuple = (output, summed_word_one_hot_vector, words_one_hot_vector_in_sequence)

    output_path = os.path.join(all_data_in_numpy_dir, id)
    pickle.dump(data_tuple, open(output_path, 'wb'))

    if i % 100 == 0:
        print ("{}/19579".format(i))



