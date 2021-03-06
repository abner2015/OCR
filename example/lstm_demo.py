"""
refector thr lstm function

"""

import mxnet as mx
import math
import time
from dataset.dataset2 import data_iter_random
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from mxnet import autograd, gluon, image, init, nd
import zipfile

from mxnet.gluon import rnn
ctx = mx.cpu()

def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

my_seq = list(range(60))
time_machine = my_seq
character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]
corpus_indices = my_seq
idx_to_char = time_machine
char_to_idx =character_dict
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01,shape=shape,ctx=ctx)
    def _three():
        return(_one((num_inputs,num_hiddens)),
                _one((num_hiddens,num_hiddens)),
               nd.zeros(num_hiddens,ctx=ctx))
    W_xi,W_hi,b_i = _three()#input gate
    W_xf,W_hf,b_f = _three() #forget gate
    W_xo,W_ho,b_o = _three() #output gate
    W_xc,W_hc,b_c = _three() #candi gate

    W_hq = _one((num_hiddens,num_outputs))
    b_q = nd.zeros(num_outputs,ctx=ctx)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

def init_lstm_state(batch_size,num_hiddens,ctx):
    return (nd.zeros(shape=(batch_size,num_hiddens),ctx=ctx),
            nd.zeros(shape=(batch_size,num_hiddens),ctx=ctx))

def lstm(inputs,state,params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        #print("lstm",X)
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    """Train an RNN model and predict the next item in the sequence."""

    data_iter_fn = data_iter_random

    #corpus_indices = my_seq
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                #print("X ",X)
                inputs = to_onehot(X, vocab_size)
                #print("len inputs ",len(inputs))
                #print("shape ",inputs)
                (outputs, state) = rnn(inputs, state, params)
                #print('len outputs',len(outputs))
                outputs = nd.concat(*outputs, dim=0)
                #print("concat output : ",len(outputs))
                #print("Y ", Y)
                y = Y.T.reshape((-1,))
                #print("y ",y)
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            sgd(params, lr, 1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            #print("n ",n)
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))

def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    """Predict next chars with a RNN model"""
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[int(prefix[0])]]
    #print("output ",output)
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    # print("idx_to_char ",idx_to_char)
    # for i in output:
    #     print("idx_to_char ",idx_to_char[i])
    return ''.join(str([idx_to_char[i] for i in output]))



if __name__ == "__main__":
    #vocab_size = 60
    import numpy as np


    loss = gluon.loss.CTCLoss()
    print(mx.nd.ones((2, 4)))
    a = mx.nd.ones((2, 4))
    b = nd.expand_dims(a,axis=0)
    print("b ",b,b.shape)
    print(mx.nd.array([[1, 0, -1, -1], [2, 1, 1, -1]]).shape)
    l = loss(mx.nd.ones((2, 30, 4)), mx.nd.array([[1, 0, -1, -1], [2, 1, 1, -1]]))
    #mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))
    print(l)
    # num_epochs, num_steps, batch_size, lr, clipping_theta = 2000, 5, 2, 1e2, 1e-2
    # pred_period, pred_len, prefixes = 8, 5, ['9', '20']
    # lstm_layer = rnn.LSTM(256)
    #
    # train_and_predict_rnn(lstm,get_params, init_lstm_state, num_hiddens,
    #                       vocab_size, ctx, corpus_indices, idx_to_char,
    #                       char_to_idx, False, num_epochs, num_steps, lr,
    #                       clipping_theta, batch_size, pred_period, pred_len,
    #                       prefixes)