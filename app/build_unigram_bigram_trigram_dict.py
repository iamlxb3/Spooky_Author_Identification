import os
import sys
import pandas as pd
import collections
import pickle


import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
sys.path.append(top_dir)

from other_funcs.nltk_funcs import preprocessing_word, tokenize_word, generate_bigrams, generate_trigrams, add_n_start



data_dir = os.path.join(top_dir, 'data')



is_validation = True
N_START = 2

if is_validation:
    train_data_path = os.path.join(data_dir, 'train_1.csv')
    language_dict_path = os.path.join(data_dir, 'language_dict_validation')
else:
    train_data_path = os.path.join(data_dir, 'train.csv')
    language_dict_path = os.path.join(data_dir, 'language_dict')

train_df = pd.read_csv(train_data_path)


# get the train unigram and bigram dict

# unigram
all_train_words_dict = collections.defaultdict(lambda :[])
for index, row in train_df.iterrows():
    text = row['text']
    author = row['author']
    word_list = tokenize_word(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = preprocessing_word(word)

    word_list = add_n_start(word_list, N_START)
    all_train_words_dict[author].extend(word_list)

for author, all_train_words in all_train_words_dict.items():
    train_unigram_dict = collections.Counter(all_train_words)
    pickle.dump(train_unigram_dict, open(os.path.join(language_dict_path, 'train_unigram_dict_{}'.
                                                      format(author)), 'wb'))
    print ('Save author-{} unigram done!'.format(author))
#

# bigram
all_train_phrases_dict = collections.defaultdict(lambda :[])
for index, row in train_df.iterrows():
    text = row['text']
    author = row['author']
    word_list = tokenize_word(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = preprocessing_word(word)

    word_list = add_n_start(word_list, N_START)
    bigrams_list = generate_bigrams(word_list)
    all_train_phrases_dict[author].extend(bigrams_list)

for author, all_train_phrases in all_train_phrases_dict.items():
    temp_all_train_phrases_dict = collections.Counter(all_train_phrases)
    #print (sorted(all_train_phrases_dict.items(), key=lambda x:x[1]))
    pickle.dump(temp_all_train_phrases_dict, open(os.path.join(language_dict_path,
                                                               'train_bigram_dict_{}'.format(author)), 'wb'))
    print ('Save author-{} bigram done!'.format(author))
#

# trigram
all_train_trigram_dict = collections.defaultdict(lambda :[])
for index, row in train_df.iterrows():
    text = row['text']
    author = row['author']
    word_list = tokenize_word(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = preprocessing_word(word)

    word_list = add_n_start(word_list, N_START)
    trigram_list = generate_trigrams(word_list)
    all_train_trigram_dict[author].extend(trigram_list)

for author, all_train_phrases in all_train_trigram_dict.items():
    temp_trigram_dict = collections.Counter(all_train_phrases)
    #print (sorted(all_train_phrases_dict.items(), key=lambda x:x[1]))
    pickle.dump(temp_trigram_dict, open(os.path.join(language_dict_path, 'train_trigram_dict_{}'.format(author)), 'wb'))
    print ('Save author-{} trigram done!'.format(author))