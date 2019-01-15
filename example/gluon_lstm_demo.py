"""
the demo is about gluon lstm train char and predict
the origin id from: https://zh.gluon.ai/chapter_recurrent-neural-networks/lstm.html
input dataset : X:[[1,2,3,4,5],[11,12,13,14,15]] Y:[[2,3,4,5,6],[12,13,14,15,16]]
"""

import mxnet as mx
import math
import time
from dataset.dataset2 import data_iter_random
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from mxnet import autograd, gluon, image, init, nd
import zipfile
import numpy as np
from mxnet.gluon import rnn,nn
ctx = mx.cpu()
from mxboard import SummaryWriter
sw = SummaryWriter(logdir='./logs', flush_secs=5)
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    """Train an Gluon RNN model and predict the next item in the sequence."""
    loss = gloss.SoftmaxCrossEntropyLoss()
    loss = gloss.CTCLoss(layout='NTC', label_layout='NT')
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter_fn = data_iter_random
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        # data_iter = data_iter_consecutive(
        #     corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        model.hybridize()
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                # X = nd.one_hot(X.T, vocab_size)
                #print(type(X))
                (output, state) = model(X,state)
                y = Y.T.reshape((-1,))
                #l = loss(output, y)
                # y = nd.one_hot(y,60)
                #model.forward(X,state)
                output = nd.expand_dims(output,axis=1)
                y = nd.expand_dims(y, axis=1)
                #print(output.shape, y.shape)
                l = loss(output, y).mean()
                # if(epoch == 0 ):
                #     sw.add_graph(model)
            l.backward()
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, ctx, idx_to_char,
                    char_to_idx))
    #model.save_params("model_lstm.params")

    model.export("gluon")
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    """Precit next chars with a Gluon RNN model"""
    state = model.begin_state(batch_size=1, ctx=ctx)

    output = [char_to_idx[int(prefix[0])]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            #print(char_to_idx)
            output.append(char_to_idx[int(prefix[t + 1])])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join(str([idx_to_char[i] for i in output]))

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
def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]
class RNNModel(gluon.HybridBlock):
    """RNN model."""
    def __init__(self, rnn,hidden_size=256,vocab_size=60, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        #self.rnn = rnn_layer

        with self.name_scope():
            self.rnn = rnn
            self.vocab_size = vocab_size
            self.dense = nn.Dense(vocab_size)

    def forward(self,inputs,state,  *args, **kwargs) :
        X = nd.one_hot(inputs.T, self.vocab_size)
        # print("db",type(self.rnn))
        Y,state = self.rnn(X, state)
        #print("dd",Y,type(Y))
        output = self.dense(Y.reshape((-1, Y.shape[-1])))

        return  output, state


    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
if __name__ == "__main__":
    #vocab_size = 60
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
    char_to_idx = character_dict
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    num_epochs, num_steps, batch_size, lr, clipping_theta = 20, 5, 2, 1e2, 1e-2
    pred_period, pred_len, prefixes = 8, 5, ['9','21']


    lstm_layer = rnn.LSTM(256)

    model = RNNModel(lstm_layer,vocab_size=vocab_size)

    train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
    model.export('gluon11')