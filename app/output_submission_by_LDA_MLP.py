import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import collections


#import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
sys.path.append(top_dir)
data_dir = os.path.join(top_dir, 'data')
from other_funcs.nltk_funcs import preprocessing_word, tokenize_word


submission_path = os.path.join(top_dir, 'submission', 'mlp_tfidf.csv')

train_data_path = os.path.join(data_dir, 'train.csv')
train_df = pd.read_csv(train_data_path)
test_data_path = os.path.join(data_dir, 'test.csv')
test_df = pd.read_csv(test_data_path)


# correct solution:
def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference




def get_X_Y(df, is_train=True):
    if is_train:
        all_labels = df['author'].values
    else:
        all_ids = df['id'].values
    all_texts = df['text'].values
    processed_texts = []
    # transform train and test
    for text in all_texts:
        word_list = tokenize_word(text)
        for i, word in enumerate(word_list):
            word = word.lower()
            word_list[i] = preprocessing_word(word)
        processed_text = ' '.join(word_list)
        processed_texts.append(processed_text)
    if is_train:
        X_Y = list(zip(processed_texts, all_labels))
        return X_Y
    else:
        X_id = list(zip(processed_texts, all_ids))
        return X_id


train_X_Y = get_X_Y(train_df)
test_X_id = get_X_Y(test_df, is_train=False)

print ("Zip X and y successfully!")


# ----------------------------------------------------------------------------------------------------------------------
# TFIDF
# ----------------------------------------------------------------------------------------------------------------------
max_n_gram = 3
max_features = 27098
token_pattern = r"[\w']+|[.,!?;]" #r"(?u)\b\w\w+\b", r"[\w']+|[.,!?;]"
tf_vectorizer = TfidfVectorizer(ngram_range=(1, max_n_gram), max_features=max_features, token_pattern=token_pattern)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# set MLP
# ----------------------------------------------------------------------------------------------------------------------
# [182, 0.00480498501684619, 235, 16, 0.0008888051045100018, 3, 1, 0.18843173771520833, 27098, 1]
# LDA_n_topics,learning_rate_init,hidden_layer_1_size,hidden_layer_2_size,alpha,max_n_gram,early_stopping,validation_fraction,max_features,token_pattern,avg_accuracy

hidden_layer_1_size = 235
learning_rate_init = 0.00480498501684619
alpha = 0.0008888051045100018
validation_fraction = 0.18843173771520833
early_stopping = True

hidden_layer_sizes = (hidden_layer_1_size, )
verbose = True
tol = 1e-5
max_iter = 1000
random_state = 999
mlp1 = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, verbose=verbose,
                     tol=tol, max_iter=max_iter, alpha=alpha, random_state=random_state, early_stopping=early_stopping,
                     validation_fraction=validation_fraction)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# set bagging classfier
# ----------------------------------------------------------------------------------------------------------------------
IS_BAGGING = False
RANDOM_STATE = 10
if IS_BAGGING:
    from sklearn.ensemble import BaggingClassifier
    mlp1 = BaggingClassifier(base_estimator=mlp1, n_estimators=5, bootstrap=True, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------------------------------------------------


# CV
# read train and test value
X_train, Y_train = list(zip(*train_X_Y))
X_test, X_ids = list(zip(*test_X_id))

# convert train X to tfidf
tf_vectorizer.fit(X_train)
X_train = tf_vectorizer.transform(X_train)
X_test = tf_vectorizer.transform(X_test)
print("Tfidf done!")


# train by classifier
mlp1.fit(X_train, Y_train)
print("Training done!")

# predict
#Y_test_pred = list(mlp1.predict(X_test))
print (mlp1.classes_)
Y_prob = list(mlp1.predict_proba(X_test))

submission_dict = collections.defaultdict(lambda :[])
for i, y_prob in enumerate(Y_prob):
    #y_prob = _softmax(y_prob)
    EAP, HPL, MWS = y_prob
    submission_dict['id'].append(X_ids[i])
    submission_dict['EAP'].append(EAP)
    submission_dict['HPL'].append(HPL)
    submission_dict['MWS'].append(MWS)



submission_df = pd.DataFrame(submission_dict, columns=['id', 'EAP', 'HPL', 'MWS'])
submission_df.to_csv(submission_path, index=False)
print ("Save submission to {} done!".format(submission_path))


