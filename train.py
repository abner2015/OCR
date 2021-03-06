
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
from mxboard import SummaryWriter

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
    batch_size = 30
    epochs = 2000
    moving_loss = 0.
    learning_rate = 0.1
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    global_step = 0
    for e in range(epochs):
        # if e == 0:
        #     net = lstm.lstm_rnn(input_sequence, h, c, temperature=temperature)
        #     sw.add_graph(lstm)
        if ((e+1) %50 == 0):
            learning_rate = learning_rate /2.0
        h = nd.zeros(shape=(batch_size,num_hidden))
        c = nd.zeros(shape=(batch_size,num_hidden))
        num_batches =2
        for i in range(num_batches):
            data_one_hot = train_data[i]
            label_one_hot = train_label[i]
            data_one_hot.attach_grad()
            label_one_hot.attach_grad()
            with autograd.record():
                outputs,h,c = lstm.lstm_rnn(inputs=data_one_hot,h=h,c=c)
                loss = lstm.average_ce_loss(outputs,label_one_hot)
                sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=global_step)
                loss.backward()
                global_step = global_step + 1


            lstm.SGD(learning_rate)
            if learning_rate % 20 == 0:
                learning_rate = learning_rate * 0.1
            if ( i == 0 ) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99*moving_loss + .01*nd.mean(loss).asscalar()

        sw.add_scalar(tag='Loss', value=moving_loss, global_step=e)
        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("1 2 3 ", 10,h,c, temperature=.1))
        print(sample("This eBook is for the use of anyone ", 10,h,c, temperature=.1))

def sample(prefix, num_chars,h,c, temperature=1.0):
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
    # h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    # c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

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