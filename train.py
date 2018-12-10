
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
from src.lstm import LSTM
from dataset.data import get_data
num_hidden = 256
with open("data/timemachine.txt") as f:
    time_machine = f.read()
time_machine = time_machine[:-38000]
character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]


def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result
def main(lstm,train_data,train_label):
    batch_size = 32
    epochs = 2000
    moving_loss = 0.
    learning_rate = 1.0

    for e in range(epochs):
        if ((e+1) %100 == 0):
            learning_rate = learning_rate /2.0
        h = nd.zeros(shape=(batch_size,num_hidden))
        c = nd.zeros(shape=(batch_size,num_hidden))
        num_batches =20
        for i in range(num_batches):
            data_one_hot = train_data[i]
            label_one_hot = train_label[i]
            data_one_hot.attach_grad()
            label_one_hot.attach_grad()
            with autograd.record():
                outputs,h,c = lstm.lstm_rnn(inputs=data_one_hot,h=h,c=c)
                loss = lstm.average_ce_loss(outputs,label_one_hot)
                loss.backward()
            lstm.SGD(learning_rate)
            if ( i == 0 ) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99*moving_loss + .01*nd.mean(loss).asscalar()
        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("The Time Ma", 1024, temperature=.1))
        print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))

def sample(prefix, num_chars, temperature=1.0):
    #####################################
    # Initialize the string that we'll return to the supplied prefix
    #####################################
    string = prefix

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    prefix_numerical = [character_dict[char] for char in prefix]
    input_sequence = one_hots(prefix_numerical)

    #####################################
    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    #####################################
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################
    for i in range(num_chars):
        outputs, h, c = lstm.lstm_rnn(input_sequence, h, c, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input_sequence = one_hots([choice])
    return string
if __name__ == '__main__':
    # with open("data/timemachine.txt") as f:
    #     time_machine = f.read()
    # time_machine = time_machine[:-3800]
    # character_list = list(set(time_machine))
    # vocab_size = len(character_list)
    ctx = mx.cpu()
    lstm = LSTM(vocab_size,ctx)
    train_data,train_label = get_data()


    main(lstm,train_data,train_label)