from nltk.stem.wordnet import WordNetLemmatizer
import nltk

lmtzr = WordNetLemmatizer()
def preprocessing_word(word):
    IS_LEMMA = True
    IS_STEMMING = False
    word = word.lower()
    if IS_LEMMA:
        word = lmtzr.lemmatize(word)
    return word

def tokenize_word(text):
    word_list = nltk.word_tokenize(text)
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