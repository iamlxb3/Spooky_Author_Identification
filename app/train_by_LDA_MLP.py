import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import pandas as pd
import collections
import pickle
from sklearn.ensemble import BaggingClassifier



#import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
sys.path.append(top_dir)
data_dir = os.path.join(top_dir, 'data')
from other_funcs.nltk_funcs import preprocessing_word, tokenize_word, add_n_start
from other_funcs.cv import create_k_fold

IS_READ_FROM_TFIDF = False
IS_LDA = False
print ("IS_READ_FROM_TFIDF: ", IS_READ_FROM_TFIDF)

if IS_READ_FROM_TFIDF:
    pass
else:
    train_data_path = os.path.join(data_dir, 'train.csv')
    train_df = pd.read_csv(train_data_path)
    all_labels = train_df['author'].values
    all_texts = train_df['text'].values
    processed_texts = []

    # transform
    for text in all_texts:
        word_list = tokenize_word(text)
        for i, word in enumerate(word_list):
            word = word.lower()
            word_list[i] = preprocessing_word(word)

        word_list = add_n_start(word_list)
        processed_text = ' '.join(word_list)
        processed_texts.append(processed_text)

    X_Y = list(zip(processed_texts, all_labels))
    print ("Zip X and y successfully!")


# ----------------------------------------------------------------------------------------------------------------------
# LDA hyper parameters
# ----------------------------------------------------------------------------------------------------------------------
def LDA_n_topics(N):
    n_topics_list = list(range(10, 200))
    for i in range(N):
        n_topics = random.sample(n_topics_list, 1)[0]
        yield n_topics


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# hyper parameters
# ----------------------------------------------------------------------------------------------------------------------

# MLP
def learning_rate_generator(N):
    for i in range(N):
        learning_rate = random.uniform(0.0001,0.001)
        yield learning_rate

def hidden_layer_1_size(N):
    for i in range(N):
        size = random.randint(200,1000)
        yield size

def hidden_layer_2_size(N):
    for i in range(N):
        size = random.randint(2, 40)
        yield size

def alpha_generator(N):
    for i in range(N):
        yield random.uniform(0.0001,0.001)

def early_stopping_generator(N):
    early_stopping_list = [True]
    for i in range(N):
        yield random.sample(early_stopping_list, 1)[0]

def validation_fraction_generator(N):
    for i in range(N):
        yield random.uniform(0.05, 0.3)

# TFIDF
def max_n_gram(N):
    max_n_gram_list = [2,3]
    for i in range(N):
        yield random.sample(max_n_gram_list, 1)[0]

def max_features_generator(N):
    for i in range(N):
        yield random.randint(20000, 40000)

def token_pattern_generator(N):
    token_pattern_list = [r"[\w']+|[.,!?;]"]
    #token_pattern_list = [r"[\w']+|[.,!?;]", r"(?u)\b\w\w+\b"]
    for i in range(N):
        yield random.sample(token_pattern_list, 1)[0]
# ----------------------------------------------------------------------------------------------------------------------

N = 15
hyper_parameter_df = []
hyper_parameter_name_list = ['LDA_n_topics','learning_rate_init','hidden_layer_1_size','hidden_layer_2_size',
                             'alpha','max_n_gram','early_stopping','validation_fraction','max_features',
                             'token_pattern',
                             'avg_accuracy']

for generator_tuple in zip(LDA_n_topics(N), learning_rate_generator(N), hidden_layer_1_size(N), hidden_layer_2_size(N),
                           alpha_generator(N), max_n_gram(N), early_stopping_generator(N),
                           validation_fraction_generator(N), max_features_generator(N), token_pattern_generator(N)):

    generator_tuple = list(generator_tuple)
    LDA_n_topics = generator_tuple[0]
    learning_rate_init = generator_tuple[1]
    hidden_layer_1_size = generator_tuple[2]
    hidden_layer_2_size = generator_tuple[3]
    alpha = generator_tuple[4]
    early_stopping = generator_tuple[6]
    if early_stopping:
        generator_tuple[6] = 1 # map True to 1
        validation_fraction = generator_tuple[7]
    else:
        generator_tuple[6] = 0 # map False to 0
        generator_tuple[7] = 0.0
        validation_fraction = 0.0

    # tfidf
    max_n_gram = generator_tuple[5]
    max_features = generator_tuple[8]
    token_pattern = generator_tuple[9]
    if token_pattern == r"[\w']+|[.,!?;]":
        generator_tuple[9] = 1 #mapping
    elif token_pattern == r"(?u)\b\w\w+\b":
        generator_tuple[9] = 2 #mapping
    else:
        raise Exception("Check token_pattern!!")

    # ----------------------------------------------------------------------------------------------------------------------
    # set MLP
    # ----------------------------------------------------------------------------------------------------------------------
    hidden_layer_sizes = (hidden_layer_1_size, )
    verbose = False
    tol = 1e-5
    max_iter = 1000
    random_state = 999
    mlp1 = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, verbose=verbose,
                         tol=tol, max_iter=max_iter, alpha=alpha, random_state=random_state, early_stopping=early_stopping,
                         validation_fraction=validation_fraction)
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # LDA
    # ----------------------------------------------------------------------------------------------------------------------
    lda = LatentDirichletAllocation(n_topics=LDA_n_topics, learning_method='online')
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # TFIDF
    # ----------------------------------------------------------------------------------------------------------------------
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, max_n_gram), max_features=max_features, token_pattern=token_pattern)
    # ----------------------------------------------------------------------------------------------------------------------

    # CV


    cv_accuracy_list = []
    tfidf_k_fold_list =[]

    if IS_READ_FROM_TFIDF:
        tfidf_k_fold_tuple = pickle.load(open('tfidf_k_fold', 'rb'))
        for train, test in tfidf_k_fold_tuple:
            X_train, Y_train = train
            X_test, Y_test = test

            if IS_LDA:
                # convert to LDA
                lda.fit(X_train)
                X_train = lda.transform(X_train)
                X_test = lda.transform(X_test)
                print("LDA done!")

            # train by classifier
            mlp1.fit(X_train, Y_train)
            print("Training done!")

            # predict
            Y_test_pred = list(mlp1.predict(X_test))
            accuracy = accuracy_score(Y_test, Y_test_pred)
            cv_accuracy_list.append(accuracy)
            print("accuracy: {}".format(accuracy))
    else:
        for train, test in create_k_fold(X_Y, k_fold=5):
            # read train and test value
            X_train, Y_train = list(zip(*train))
            X_test, Y_test = list(zip(*test))

            # convert train X to tfidf
            tf_vectorizer.fit(X_train)
            X_train = tf_vectorizer.transform(X_train)
            X_test = tf_vectorizer.transform(X_test)
            print("Tfidf done!")

            # tfidf_k_fold_list.append(((X_train, Y_train), (X_test, Y_test)))
            # continue

            if IS_LDA:
                # convert to LDA
                #lda = LatentDirichletAllocation(n_topics=LDA_n_topics, learning_method='online')
                lda.fit(X_train)
                X_train = lda.transform(X_train)
                X_test = lda.transform(X_test)
                print("LDA done!")

            # train by classifier
            mlp1.fit(X_train, Y_train)
            print("Training done!")

            # predict
            Y_test_pred = list(mlp1.predict(X_test))
            accuracy = accuracy_score(Y_test, Y_test_pred)
            cv_accuracy_list.append(accuracy)
            print ("accuracy: {}".format(accuracy))



    avg_accuracy = np.average(cv_accuracy_list)
    hyper_parameter_value_list = list(generator_tuple) + [avg_accuracy]
    hyper_parameter_df.append(dict(zip(hyper_parameter_name_list, hyper_parameter_value_list)))

    print ("------------------------------------")
    print ("generator_tuple: ", generator_tuple)
    print ("avg_accuracy: {}".format(avg_accuracy))

    #pickle.dump(tuple(tfidf_k_fold_list), open('tfidf_k_fold', 'wb'))

    #sys.exit()


# LDA_n_topics = generator_tuple[0]
# learning_rate_init = generator_tuple[1]
# hidden_layer_1_size = generator_tuple[2]
# hidden_layer_2_size = generator_tuple[3]
# alpha = generator_tuple[4]
# early_stopping = generator_tuple[6]
# validation_fraction = generator_tuple[7]
# # tfidf
# max_n_gram = generator_tuple[5]
# max_features = generator_tuple[8]

hyper_parameter_df = pd.DataFrame(hyper_parameter_df, columns=hyper_parameter_name_list)
hyper_parameter_df = hyper_parameter_df.sort_values('avg_accuracy', ascending=False)
print (hyper_parameter_df)
hyper_parameter_df.to_csv('hyper_parameter_df.csv', index=False, )


