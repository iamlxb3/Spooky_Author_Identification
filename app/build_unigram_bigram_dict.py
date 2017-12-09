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

from other_funcs.nltk_funcs import lemmatize_word, tokenize_word, generate_bigrams



data_dir = os.path.join(top_dir, 'data')
train_data_path = os.path.join(data_dir, 'train_1.csv')

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
        word_list[i] = lemmatize_word(word)

    word_list = ['start_0'] + word_list + ['end_0']
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
    word_list = tokenize_word(text)
    for i, word in enumerate(word_list):
        word = word.lower()
        word_list[i] = lemmatize_word(word)
    word_list = ['start_0'] + word_list + ['end_0']
    bigrams_list = generate_bigrams(word_list)
    all_train_phrases_dict[author].extend(bigrams_list)

for author, all_train_phrases in all_train_phrases_dict.items():
    all_train_phrases_dict = collections.Counter(all_train_phrases)
    #print (sorted(all_train_phrases_dict.items(), key=lambda x:x[1]))
    pickle.dump(all_train_phrases_dict, open(os.path.join(data_dir, 'language_dict', 'train_bigram_dict_{}'.format(author)), 'wb'))
    print ('Save author-{} bigram done!'.format(author))
#