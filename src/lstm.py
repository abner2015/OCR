from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np

class LSTM(object):
    def __init__(self,vocab_size,ctx):
        self.num_inputs = vocab_size
        self.num_hidden = 256
        self.num_outputs = vocab_size

        ########################
        #  Weights connecting the inputs to the hidden layer
        ########################
        self.Wxg = nd.random_normal(shape=(self.num_inputs, self.num_hidden), ctx=ctx) * .01
        self.Wxi = nd.random_normal(shape=(self.num_inputs, self.num_hidden), ctx=ctx) * .01
        self.Wxf = nd.random_normal(shape=(self.num_inputs, self.num_hidden), ctx=ctx) * .01
        self.Wxo = nd.random_normal(shape=(self.num_inputs, self.num_hidden), ctx=ctx) * .01

        ########################
        #  Recurrent weights connecting the hidden layer across time steps
        ########################
        self.Whg = nd.random_normal(shape=(self.num_hidden, self.num_hidden), ctx=ctx) * .01
        self.Whi = nd.random_normal(shape=(self.num_hidden, self.num_hidden), ctx=ctx) * .01
        self.Whf = nd.random_normal(shape=(self.num_hidden, self.num_hidden), ctx=ctx) * .01
        self.Who = nd.random_normal(shape=(self.num_hidden, self.num_hidden), ctx=ctx) * .01

        ########################
        #  Bias vector for hidden layer
        ########################
        self.bg = nd.random_normal(shape=self.num_hidden, ctx=ctx) * .01
        self.bi = nd.random_normal(shape=self.num_hidden, ctx=ctx) * .01
        self.bf = nd.random_normal(shape=self.num_hidden, ctx=ctx) * .01
        self.bo = nd.random_normal(shape=self.num_hidden, ctx=ctx) * .01

        ########################
        # Weights to the output nodes
        ########################
        self.Why = nd.random_normal(shape=(self.num_hidden, self.num_outputs), ctx=ctx) * .01
        self.by = nd.random_normal(shape=self.num_outputs, ctx=ctx) * .01

        self.params = [self.Wxg, self.Wxi, self.Wxf, self.Wxo, self.Whg, self.Whi, self.Whf, self.Who, self.bg, self.bi, self.bf, self.bo, self.Why, self.by]
        for param in self.params:
            param.attach_grad()
    def lstm_rnn(self,inputs,h,c,temperature=1.0):
        outputs = []
        for X in inputs:
            # if not X.shape[0] == 77:
            #     continue
            #print("X.shape",X.shape,self.Wxg.shape)
            g = nd.tanh(nd.dot(X, self.Wxg)+nd.dot(h,self.Whg)+self.bg)
            i = nd.sigmoid(nd.dot(X,self.Wxi)+nd.dot(h,self.Whi)+self.bi)
            f = nd.sigmoid(nd.dot(X,self.Wxf)+nd.dot(h,self.Whf)+self.bf)
            o = nd.sigmoid(nd.dot(X,self.Wxo)+nd.dot(h,self.Who)+self.bo)

            c = f*c + i*g
            h = o*nd.tan(c)

            yhat_linear = nd.dot(h,self.Why) + self.by
            #yhat = self.softmax(yhat_linear,temperature=temperature)
            yhat = mx.ndarray.softmax(yhat_linear,temperature=temperature)
            outputs.append(yhat)
        return (outputs,h,c)
    def softmax(self,y_linear,temperature=1.0):
        lin = (y_linear - nd.max(y_linear)) / temperature
        exp = nd.exp(lin)
        partition = nd.sum(exp,axis=0,exclude=True).reshape((-1,1))
        return exp / partition

    def cross_entropy(self,yhat,y):
        return - nd.mean(nd.sum(y*nd.log(yhat),axis=0,exclude=True))

    def average_ce_loss(self,outputs,labels):
        #print(len(outputs),len(labels))
        assert (len(outputs) == len(labels))
        total_loss = 0.
        for (output,label) in zip(outputs,labels):
            total_loss = total_loss + self.cross_entropy(output,label)
        return total_loss / len(outputs)

    def SGD(self,lr):
        for param in self.params:
            param[:] = param - lr*param.grad