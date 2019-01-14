from mxnet import gluon
from mxnet .gluon import nn ,rnn
class OCRLSTM(nn.HybridBlock):
    def __init__(self,hidden_size=256,num_layers=1,vocab_size=4):
        super(OCRLSTM,self).__init__()
        self.num_hidden = hidden_size
        with self.name_scope():
            self.cnn = nn.Conv2D(channels=1,kernel_size=3,layout="NCHW")
            self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size,num_layers=num_layers,layout="NTC")
            self.fc = nn.Dense(vocab_size,in_units=hidden_size,flatten=False)
            #self.state = self.rnn.begin_state(batch_size=22)
    def hybrid_forward(self, F,inputs,state):
        #print("input",inputs.shape)
        X = self.cnn(inputs)
        X = X.squeeze(axis=1)
        # print(X.shape)
        X = X.transpose((0,2,1))
        print("transpose",type(X))
        #print(X.shape,state.shape)
        X,state = self.rnn(X,state)
        print("type",type(X))
        # print("rnn",X.shape)
        #print("rnn", X.shape)
        #X = X.reshape((-1, X.shape[-1]))
        X = self.fc(X.reshape((-1, self.num_hidden)))
        #print("fc", X.shape)
        return X,state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
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

    loss = gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    trainer = gluon.Trainer(lstm.collect_params(),'sgd',{'learning_rate':0.001})
    state = lstm.begin_state(batch_size=1)
    global_step = 0


    # for epoch in range(100):
    #     train_loss = .0
    #     with autograd.record():
    #         #print("data ",state)
    #         output,state = lstm(data,state)
    #         # s = net.tojson()
    #         # print(s)
    #         # net.export("model10")
    #         #output = output.transpose((2,1,0))
    #
    #         output = nd.expand_dims(output, axis=1)
    #         output = nd.transpose(output,(2,1,0))
    #         #label = nd.expand_dims(label, axis=1)
    #         print("output ",output.shape,label.shape)
    #         L = loss(output,label)
    #     L.backward()
    #     train_loss = nd.mean(L).asscalar()
    #     sw.add_scalar(tag="loss",value=train_loss,global_step=global_step)
    #     global_step = global_step + 1
    #     if epoch == 1 :
    #         sw.add_graph(lstm)
    #     trainer.step(1)
    #     lstm.save_parameters("mo1.params")
    #     if(epoch %100 == 0):
    #         print('train_loss %.4f'%(train_loss))
    #         # print('output max', output.argmax(axis=2))
    # #export the model
    # # net.save_params()
    # lstm.export("mod1")
    net = gluon.nn.SymbolBlock.imports('mod1-symbol.json', ['data0'], param_file='mod1-0000.params',   ctx=mx.cpu())
    output = net(data)
    output = output[0].reshape((-1, output.shape[-1]))
    print (output.shape )
    print(output.argmax(axis=1))
    print(len(output.argmax(axis=1)))

