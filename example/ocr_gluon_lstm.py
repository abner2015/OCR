from mxnet import gluon,nd
from mxnet .gluon import nn ,rnn

class OCRLSTM(nn.Block):
    def __init__(self,hidden_size=256,num_layers=1,vocab_size=11):
        super(OCRLSTM,self).__init__()
        self.num_hidden = hidden_size
        with self.name_scope():
            self.cnn = nn.Conv2D(channels=1,kernel_size=3,layout="NCHW")
            self.polling = nn.AvgPool2D(pool_size=(3,3),strides=1)
            self.act = nn.Activation(activation='relu')
            self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size,num_layers=num_layers,layout="NTC")
            self.fc = nn.Dense(vocab_size,in_units=hidden_size,flatten=False)
            #self.state = self.rnn.begin_state(batch_size=22)
    def forward(self, inputs,state, *args, **kwargs):
        print("input",inputs.shape)
        # input shape 1*3*36*248 (batch_size, in_channels, height, width)`
        X = self.cnn(inputs)
        X = self.polling(X)
        X = self.act(X)
        #X = X.squeeze(axis=1)
        print("cnn",X.shape)
        #X = X.resize((1,1,9,62))
        X = X.transpose((3,0,1,2))
        print("transpose",X.shape)
        #print(X.shape,state.shape)
        output = []
        for time_step in X:
            #print("time_step")
            X,state = self.rnn(time_step,state)
            output.append(X)
        print("type",type(X))
        print("rnn",X.shape)
        #print("rnn", X.shape)
        #X = X.reshape((-1, X.shape[-1]))
        X = mx.nd.concat(*output,dim=0)
        print("concat shape ",X.shape)
        X = self.fc(X)
        #X = self.fc(X.reshape((-1, self.num_hidden)))
        print("fc", X.shape)

        return X,state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
# -1 is background
char_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',-1:'_'}
id2char = [0,1,2,3,4,5,6,7,8,9,'_']
vocab_size = len(char_dict)
char2id = []
# X = nd.array([[6],[0],[8],[1]])
# a =[nd.one_hot(x, vocab_size) for x in X.T]
# a = nd.array(a[0])
# print("aaaa ",a,)
def ctc_label(p):
    """
    Iterates through p, identifying non-zero and non-repeating values, and returns them in a list
    Parameters
    ----------
    p: list of int
    Returns
    -------
    list of int
    """
    ret = []
    p1 = [0] + p
    for i, _ in enumerate(p):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret
def to_ctc_format(label,seq_length):
    """

    :param label: str "0123"
    :param seq_length: equal the time_step
    :return: [0,1,2,3,-1,-1,...]
    """
    #seq_length = 176
    str = label
    str_list = [x for x in str]
    #print('str_list',str_list)
    index = 0
    length = (len(str_list))
    label_list = []
    # add -1 at repeat num ,such as : 00-->0-*-0
    while index < length:
        #print (index)
        if index + 1 < length and str_list[index] == str_list[index+1] :
            label_list.append(int(str_list[index]))
            label_list.append(-1)
        else:
            label_list.append(int(str_list[index]))
        index = index + 1
    #print(label_list)
    if len(label_list) < seq_length:
        le = seq_length - len(label_list)
        other = [-1]*le
        print("other ",other)
        label_list.extend(other)
    #print (label_list)
    return label_list
def data_iter(img_path,label_str):
    """

    :param img_path:  1.jpg
    :param label_str: 2345
    :return: data,label
    """
    data = mx.image.imread(img_path)
    data = data.astype('float32') / 255.0
    data = data - 0.5
    print("data shape ", data.shape)
    data = data.reshape(1, 128, 60, 3)
    print("data shape ", data.shape)
    data = nd.transpose(data, (0, 3, 1, 2))

    print("after data shape :", data.shape)
    # ch2id = nd.array([[0,0,0],[0],[8],[1]])
    #label_str = '0681'
    label_list = to_ctc_format(label_str, 56)
    print("label_list",label_list)
    label = nd.array([label_list])

    return data,label
if __name__ == "__main__":

    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    from mxnet import autograd, gluon, image, init, nd
    import mxnet as mx
    img_path = ".././data/bo681.jpg"
    label_str = '0681'
    data,label = data_iter(img_path,label_str)

    lstm = OCRLSTM()
    lstm.collect_params().initialize(mx.init.Xavier(),ctx = mx.cpu())

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
            #output = nd.expand_dims(output, axis=1)
            output = output.transpose((1, 0, 2))
            #label = nd.expand_dims(label, axis=1)
            #label = label.reshape((1,4))
            print("output ",output.shape,label.shape)
            L = loss(output,label)
        L.backward()
        train_loss = nd.mean(L).asscalar()
        #sw.add_scalar(tag="loss",value=train_loss,global_step=global_step)
        global_step = global_step + 1
        # if epoch == 1 :
        #     sw.add_graph(net)
        trainer.step(1,ignore_stale_grad=True)

        if(epoch %100 == 0):
            print('train_loss %.4f'%(train_loss))
            # print('output max', output.argmax(axis=2))
        op,state = lstm(data,state)
        print('outpuddd', op.shape)
        #print('outpuddd', op)
        op = nd.reshape(op,(56,11))
        #print("dddtest ",test)
        #print('op',op.shape)
        #print(op[0].asnumpy())
        #op = op[0].asnumpy()
        tt = mx.nd.softmax(op)
        tt2 = tt.asnumpy().argmax(axis=1)
        print ("op tt ",tt.asnumpy().argmax( axis=1))
        print("length :",len(tt.asnumpy().argmax( axis=1)))
        tag = [ d for d in tt2]
        #rec = ctc_label(tt.asnumpy().argmax( axis=1))
        #print("tag : ",tag)
        str_num = ''
        for i in tag:
            #if i in char_dict.keys():
            str_num = str_num + " "+str(id2char[i])
        print("result ",str_num)
        #prediction = [p - 1 for p in rec]
        # print("prediction : ",prediction)

    #export the model
    net.save_params("test")
    #net.export("mod1")
    # net = gluon.nn.SymbolBlock.imports('mod1-symbol.json', ['data'], param_file='mod1-0000.params',   ctx=mx.cpu())
    # output = net(data)
    # output = output[0].reshape((-1, output.shape[-1]))
    # print (output.shape )
    # print(output.argmax(axis=1))
    # print(len(output.argmax(axis=1)))

