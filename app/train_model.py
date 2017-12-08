import os
import sys
import pickle
import torch
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import io
import collections
import torch.nn as nn
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
sys.path.append(top_dir)

from model.rnn_pytorch import RNNPytorch

data_dir = os.path.join(top_dir, 'data')
all_data_in_numpy_dir = os.path.join(data_dir, 'all_data_in_numpy')


# ----------------------------------------------------------------------------------------------------------------------
# split train, test
# ----------------------------------------------------------------------------------------------------------------------
sentence_id_list = os.listdir(all_data_in_numpy_dir)
train_sentence_id_list = sentence_id_list[0:14000]
validation_sentence_id_list = sentence_id_list[14000:]
train_sentence_id_paths = [os.path.join(all_data_in_numpy_dir, x) for x in train_sentence_id_list]
validation_sentence_id_paths = (os.path.join(all_data_in_numpy_dir, x) for x in validation_sentence_id_list)
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# build RNN
# ----------------------------------------------------------------------------------------------------------------------
n_hidden = 20
input_size = 22453
output_size = 3
learning_rate = 0.005
rnn = RNNPytorch(input_size, n_hidden, output_size, learning_rate)
Loss = nn.NLLLoss()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# train functions
# ----------------------------------------------------------------------------------------------------------------------

def train(rnn, category_tensor, line_tensor):
    learning_rate = rnn.learning_rate
    hidden = rnn.initHidden()

    rnn.zero_grad()

    sequence_length = line_tensor.size()[0]
    for i in range(sequence_length):
        if i < sequence_length - 1:
            hidden = rnn(line_tensor[i], hidden)
        elif i == sequence_length - 1:
            output = rnn.get_output(line_tensor[i], hidden)

    loss = Loss(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Just return an output given a line
# ----------------------------------------------------------------------------------------------------------------------
def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------------------------
MAX_ITER = 20000

current_loss = 0
print_every = 40
loss_plot_list = []

def _plot(plot_list):
    plt.plot(plot_list)
    plt.show()
    plt.savefig('loss.png')

for i in range(MAX_ITER):
    train_sample_path = random.sample(train_sentence_id_paths, 1)[0]
    f = pickle.load(open(train_sample_path, 'rb'))
    # output
    output, summed_word_one_hot_vector, words_one_hot_vector_in_sequence = f
    output = list(output).index(1) # convert np.array([1,0,0] to 1
    actual_output = torch.autograd.Variable(torch.LongTensor([output]))
    # one-hot-vector-in-sequence
    input = torch.from_numpy(words_one_hot_vector_in_sequence)
    input = input.view(input.size()[0], 1, -1)
    input = input.type(torch.FloatTensor)
    input = torch.autograd.Variable(input)


    # train start
    pred_output, loss = train(rnn, actual_output, input)
    current_loss += loss

    if i!= 0 and i % print_every == 0:
        print ("iter-{}, current_loss: {}".format(i, current_loss))
        loss_plot_list.append(current_loss)
        current_loss = 0


_plot(loss_plot_list)

pickle.dump(rnn, open(os.path.join(top_dir,'trained_model', 'rnn'), 'wb'))

    # print (actual_output)
    # print (input)
    # sys.exit()
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------------------------------

# load rnn
rnn = pickle.load(open(os.path.join(top_dir,'trained_model','rnn'), 'rb'))
#

pred_label_list = []
actual_label_list = []

for i, id_path in enumerate(validation_sentence_id_paths):
    sample_path = random.sample(train_sentence_id_paths, 1)[0]
    f = pickle.load(open(sample_path, 'rb'))

    # output
    output, summed_word_one_hot_vector, words_one_hot_vector_in_sequence = f
    output = list(output).index(1) # convert np.array([1,0,0] to 1
    actual_output = torch.autograd.Variable(torch.LongTensor([output]))

    # one-hot-vector-in-sequence
    input = torch.from_numpy(words_one_hot_vector_in_sequence)
    input = input.view(input.size()[0], 1, -1)
    input = input.type(torch.FloatTensor)
    input = torch.autograd.Variable(input)

    pred_output = evaluate(rnn, input)
    pred_output = pred_output[0].max(0)[1].data[0]

    pred_label_list.append(pred_output)
    actual_label_list.append(output)

    if i % 20 == 0:
        print ("validation-sample-{}".format(i))



accuracy = accuracy_score(actual_label_list, pred_label_list)
print ("Accuracy: {}".format(accuracy))
print (collections.Counter(pred_label_list))
# ----------------------------------------------------------------------------------------------------------------------


















