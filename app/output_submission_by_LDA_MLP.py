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
test_X_id = get_X_Y(train_df, is_train=False)

print ("Zip X and y successfully!")



# ----------------------------------------------------------------------------------------------------------------------
# set MLP
# ----------------------------------------------------------------------------------------------------------------------

hidden_layer_1_size = 897
learning_rate_init = 0.00896
alpha = 0.00029
validation_fraction = 0.1
early_stopping = False

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
# TFIDF
# ----------------------------------------------------------------------------------------------------------------------
max_n_gram = 2
max_features = 3983
token_pattern = r"(?u)\b\w\w+\b" #r"(?u)\b\w\w+\b", r"[\w']+|[.,!?;]"
tf_vectorizer = TfidfVectorizer(ngram_range=(1, max_n_gram), max_features=max_features, token_pattern=token_pattern)
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
    y_prob = _softmax(y_prob)
    EAP, HPL, MWS = y_prob
    submission_dict['id'].append(X_ids[i])
    submission_dict['EAP'].append(EAP)
    submission_dict['HPL'].append(HPL)
    submission_dict['MWS'].append(MWS)



submission_df = pd.DataFrame(submission_dict, columns=['id', 'EAP', 'HPL', 'MWS'], index=False)
submission_df.to_csv(submission_path)



