import os
import sys
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')

skip_gram_output_dict_path = os.path.join(data_dir, 'skip_gram_output_dict')
train_word_dict_0_count_path = os.path.join(data_dir, 'train_word_dict_0_count')
word_index_dict_path = os.path.join(data_dir, 'word_index_dict')

train_word_dict_0_count = pickle.load(open(train_word_dict_0_count_path, 'rb'))
skip_gram_output_dict = pickle.load(open(skip_gram_output_dict_path, 'rb'))

# ordered_words = sorted(train_word_dict_0_count.keys())
# word_index_dict = {word:ordered_words.index(word) for word in ordered_words}
# pickle.dump(word_index_dict, open(word_index_dict_path, 'wb'))
# sys.exit()

word_index_dict = pickle.load(open(word_index_dict_path, 'rb'))
word_total_num = len(word_index_dict)

# build MLP classifier
word_embedding_num = 20
MLP1 = MLPRegressor(hidden_layer_sizes=(word_embedding_num,),
                    random_state=1, learning_rate_init=0.01,
                    verbose=True, tol = 1e-6, max_iter=2000)



for word, output in skip_gram_output_dict.items():

    # get the input for MLP
    one_hot_word_vector = np.zeros(word_total_num)
    word_index = word_index_dict[word]
    one_hot_word_vector[word_index] += 1
    one_hot_word_vector = one_hot_word_vector.reshape(1,-1)
    output = output.reshape(1,-1)
    MLP1.fit(one_hot_word_vector, output)
    sys.exit()
    #