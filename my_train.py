
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
from mxboard import SummaryWriter
import dataset.dataset2 as ds
mx.random.seed(1)
from src.lstm_old import LSTM
import numpy as np
from dataset.data import get_data
num_hidden = 256
# with open("data/timemachine.txt") as f:
#     time_machine = f.read()
my_seq = list(range(60))
print("my_seq ",my_seq)
#time_machine = time_machine[:-38000]
time_machine = my_seq
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
    batch_size = 5
    epochs = 2000
    moving_loss = 0.
    learning_rate = 1.0
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
        dataset = ""
        for X, Y in ds.data_iter_random(my_seq, batch_size=5, num_steps=5):
            #print("orgil",X)
            #b = X.reshape((num_batches, batch_size, seq_length, vocab_size))
            #X = nd.one_hot(X, vocab_size)
            X = X.T

            #print(X)
            #Y = nd.one_hot(Y, vocab_size)
            data_one_hot = X
            label_one_hot = Y
            #print(X.shape,Y)

            #Y = [nd.one_hot(x, 60) for x in Y]
            label_one_hot =  Y.T.reshape((-1,))
            #print(Y)
            data_one_hot.attach_grad()
            #label_one_hot.attach_grad()
            with autograd.record():
                outputs,h,c = lstm.lstm_rnn(inputs=data_one_hot,h=h,c=c)
                #print("type",outputs)
                outputs = nd.concat(*outputs, dim=0)
                loss = lstm.average_ce_loss(outputs,label_one_hot)
                sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=global_step)
                global_step = global_step + 1
            loss.backward()



            lstm.SGD(learning_rate)
            if learning_rate % 200 == 0:
                learning_rate = learning_rate * 0.1
            if  (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99*moving_loss + .01*nd.mean(loss).asscalar()

        sw.add_scalar(tag='Loss', value=moving_loss, global_step=e)
        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("1", 5,h,c, temperature=.1))
        #print(sample("This eBook is for the use of anyone ", 10,h,c, temperature=.1))

def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
def to_onehot(X, size):  # 本函数已保存在 d2lzh 包中方便以后使用。
    return (nd.one_hot(x, size) for x in X.T)
def sample(prefix, num_chars,h,c, temperature=1.0):
    #####################################
    # Initialize the string that we'll return to the supplied prefix
    #####################################
    string = prefix+" "

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    #print( character_dict,prefix)
    # for char in prefix :
    #     print (char)
    prefix_numerical = [character_dict[int(char)] for char in prefix]
    input_sequence =  prefix_numerical
    input_sequence = nd.array(input_sequence)
    #input_sequence = (nd.one_hot(x , 60) for x  in input_sequence.T)
    input_sequence = one_hots([int(prefix)],60)

    #input_sequence = one_hots(prefix_numerical)

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
        #print("choice",choice)
        string += str(character_list[choice])+" "
        input_sequence = one_hots([choice])
    return string
if __name__ == '__main__':
    # with open("data/timemachine.txt") as f:
    #     time_machine = f.read()
    # time_machine = time_machine[:-3800]
    # character_list = list(set(time_machine))
    # vocab_size = len(character_list)
    ctx = mx.cpu()
    lstm = LSTM(30,ctx)
    # train_data,train_label = get_data()
    train_data, train_label = " "," "

    main(lstm,train_data,train_label)