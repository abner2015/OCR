from mxnet import gluon
from mxnet .gluon import nn ,rnn
class OCRLSTM(nn.HybridBlock):
    def __init__(self,hidden_size=256,num_layers=1,vocab_size=4):
        super(OCRLSTM,self).__init__()
        self.num_hidden = hidden_size
        with self.name_scope():
            self.cnn = nn.Conv2D(channels=1,kernel_size=3,layout="NCHW")
            self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size,num_layers=num_layers,layout="NTC")
            self.fc = nn.Dense(vocab_size,in_units=hidden_size,flatten=False,prefix='ocr_')
            #self.state = self.rnn.begin_state(batch_size=22)
    def hybrid_forward(self, F,inputs,state):
        #print("input",inputs.shape)
        # input shape 1*3*36*248 (batch_size, in_channels, height, width)`
        X = self.cnn(inputs)
        X = F.reshape(X,shape=[1,1,178,48])
        print("cnn",type(X))
        X = X.squeeze(axis=1)
        #print(X.shape)
        X = X.transpose((1,2,0))
        print("infer",self.infer_shape(X))
        for x in F.array(X):
            print("dtest : ",x)
        print("transpose",len(X))
        #print(X.shape,state.shape)
        X,state = self.rnn(X,state)
        print("type",type(state))
        # print("rnn",X.shape)
        #print("rnn", X.shape)
        #X = X.reshape((-1, X.shape[-1]))
        X = self.fc(X.reshape((-1, self.num_hidden)))
        #print("fc", X.shape)
        return X,state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
def predict(lstm,data,state):
    char_to_idx = [6,0,8,1]
    state = lstm.begin_state(batch_size=1, ctx=mx.cpu())
    prefix = data
    #output = [char_to_idx[int(prefix[0])]]
    num_chars = 4
    # for t in range(num_chars + len(prefix) - 1):
    #     X = nd.array([output[-1]], ctx=mx.cpu()).reshape((1, 1))
    #     Y, state = lstm(data, state)
    #     Y = nd.expand_dims(Y, axis=1)
    #     Y = nd.transpose(Y, (2, 1, 0))
    #     #(Y, state) = model(X, state)
    #     if t < len(prefix) - 1:
    #         # print(char_to_idx)
    #         output.append(char_to_idx[int(prefix[t + 1])])
    #     else:
    #         output.append(int(Y.argmax(axis=1).asscalar()))
    #return ''.join(str([idx_to_char[i] for i in output]))
if __name__ == "__main__":
    from mxboard import SummaryWriter

    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    from mxnet import autograd, gluon, image, init, nd
    import mxnet as mx
    data = mx.image.imread("../data/bo681.jpg")
    data = data.astype('float32')/255.0
    data = data-0.5
    print("data shape ",data.shape)
    data = data.reshape(1,50, 180, 3)
    print("data shape ", data.shape)
    data = nd.transpose(data.reshape(1,50,180,3),(0,3,1,2))
    print("after data shape :",data.shape)
    label = nd.array([[6],[0],[8],[1]])
    #net = nn.HybridSequential()
    lstm = OCRLSTM(vocab_size=4)
    lstm.collect_params().initialize(mx.init.Xavier(),ctx = mx.cpu())

    # with net.name_scope():
    #     net.add(lstm)
    # net.initialize()
    # net.hybridize()
    #print(net)
    lstm.initialize()
    lstm.hybridize()
    #print(lstm)
    loss = gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    trainer = gluon.Trainer(lstm.collect_params(),'sgd',{'learning_rate':0.001})
    state = lstm.begin_state(batch_size=1)
    global_step = 0


    for epoch in range(100):
        train_loss = .0
        with autograd.record():
            #print("data ",state)
            output,state = lstm(data,state)
            # s = net.tojson()
            # print(s)
            # net.export("model10")
            #output = output.transpose((2,1,0))

            output = nd.expand_dims(output, axis=1)
            output = nd.transpose(output,(2,1,0))
            #label = nd.expand_dims(label, axis=1)
            #print("output ",output.shape,label.shape)
            L = loss(output,label)
        L.backward()
        train_loss = nd.mean(L).asscalar()
        sw.add_scalar(tag="loss",value=train_loss,global_step=global_step)
        global_step = global_step + 1
        if epoch == 1 :
            sw.add_graph(lstm)
        trainer.step(1)
        lstm.save_parameters("mo1.params")
        if(epoch %100 == 0):
            print('train_loss %.4f'%(train_loss))
            # print('output max', output.argmax(axis=2))

        #print(" result ",predict(lstm,data,state))
    #export the model

    lstm.export("mod1")
    net = gluon.nn.SymbolBlock.imports('mod1-symbol.json', ['data0'],    ctx=mx.cpu())
    net.initialize()
    net.hybridize()
    # net.load_parameters(lstm.begin_state())
    #net = mx.nd.load("mod1-0000.params")
    #state = lstm.begin_state(1)
    print(net)
    output = net(data)
    import numpy

    output = numpy.array(output)
    # print(output[0].reshape(-1,output.shape()).shape)

    output = output[0].reshape((-1, output.shape[-1]))
    print (output.shape )
    print(output.argmax(axis=1))
    print(len(output.argmax(axis=1)))

