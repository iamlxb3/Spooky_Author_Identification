from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))

lmtzr = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

# get stopwords
stopwords_set = set()
with open (os.path.join(current_path, 'english_stopwords.txt'), 'r') as f:
    for line in f:
        word = line.strip()
        stopwords_set.add(word)

def preprocessing_word(word):
    IS_LEMMA = True
    IS_STEMMING = True
    word = word.lower()
    if IS_LEMMA:
        word = lmtzr.lemmatize(word)
    if IS_STEMMING:
        word = porter_stemmer.stem(word)
    return word

def tokenize_word(text):
    IS_FILTER_STOPWORD = False
    word_list = nltk.word_tokenize(text)
    if IS_FILTER_STOPWORD:
        new_word_list = []
        for word in word_list:
            if word not in stopwords_set:
                new_word_list.append(word)
        word_list = new_word_list
    return word_list


def generate_bigrams(word_list):
    bigrams_list = []
    for i, word in enumerate(word_list):
        if i <= len(word_list) - 2:
            phrases = (word, word_list[i+1])
            bigrams_list.append(phrases)
    return bigrams_list

def generate_trigrams(word_list):
    trigrams_list = []
    for i, word in enumerate(word_list):
        if i <= len(word_list) - 3:
            phrases = (word, word_list[i+1], word_list[i+2])
            trigrams_list.append(phrases)
    return trigrams_list