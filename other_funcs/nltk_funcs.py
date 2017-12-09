from nltk.stem.wordnet import WordNetLemmatizer
import nltk

lmtzr = WordNetLemmatizer()
def lemmatize_word(word):
    IS_LEMMA = True
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