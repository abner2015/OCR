from mxnet import gluon
from mxnet .gluon import nn ,rnn
class OCRLSTM(nn.Block):
    def __init__(self,hidden_size=256,num_layers=1,vocab_size=4):
        super(OCRLSTM,self).__init__()
        with self.name_scope():
            self.cnn = nn.Conv2D(channels=1,kernel_size=3,layout="NCHW")
            self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size,num_layers=num_layers,layout="NTC")
            self.fc = nn.Dense(units=vocab_size,flatten=False,activation="sigmoid")
            self.state = self.rnn.begin_state(batch_size=22)
    def forward(self, inputs, *args, **kwargs):
        #print(x.shape)

        X = self.cnn(inputs)
        X = X.squeeze(axis=1)
        #print(X.shape)
        X = X.transpose((0,2,1))
        print("transpose",type(X))
        print(X.shape)
        X,self.state = self.rnn(X,self.state)
        print("rnn",X.shape)
        #print("rnn", X.shape)
        #X = X.reshape((-1, X.shape[-1]))
        X = self.fc(X)
        #print("fc", X.shape)
        return X

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
    data = nd.transpose(data.reshape(-1,50,180,3),(0,3,1,2))
    label = nd.array([[6],[0],[8],[1]])
    net = nn.Sequential()
    lstm = OCRLSTM()
    state = lstm.begin_state()
    with net.name_scope():
        net.add(lstm)
    net.initialize()
    net.hybridize()
    #print(net)
    loss = gluon.loss.CTCLoss()
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.001})
    global_step = 0


    for epoch in range(100):
        train_loss = .0
        with autograd.record():

            output = net(data)
            # s = net.tojson()
            # print(s)
            # net.export("model10")
            output = output.transpose((2,1,0))

            # output = output.squeeze(axis=1)
            #print("output ",output.shape,label.shape)
            L = loss(output,label)
        L.backward()
        train_loss = nd.mean(L).asscalar()
        sw.add_scalar(tag="loss",value=train_loss,global_step=global_step)
        global_step = global_step + 1
        if epoch == 1 :
            sw.add_graph(net)
        trainer.step(1)
        net.save_parameters("mo1.params")
        if(epoch %100 == 0):
            print('train_loss %.4f'%(train_loss))
            # print('output max', output.argmax(axis=2))
    #export the model
    net.export("mod1")
    # net = gluon.nn.SymbolBlock.imports('mod1-symbol.json', ['data'], param_file='mod1-0000.params',   ctx=mx.cpu())
    # output = net(data)
    # output = output[0].reshape((-1, output.shape[-1]))
    # print (output.shape )
    # print(output.argmax(axis=1))
    # print(len(output.argmax(axis=1)))

