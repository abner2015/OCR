from mxnet import gluon,nd
from mxnet .gluon import nn ,rnn
import os
class OCRLSTM(nn.Block):
    def __init__(self,hidden_size=256,num_layers=2,vocab_size=11):
        super(OCRLSTM,self).__init__()
        self.num_hidden = hidden_size
        with self.name_scope():
            self.cnn = nn.Conv2D(channels=1,kernel_size=3,layout="NCHW")
            self.polling = nn.AvgPool2D(pool_size=(3,3),strides=1)
            #self.act = nn.Activation(activation='relu')

            self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size,num_layers=num_layers,layout="NTC")
            self.fc = nn.Dense(vocab_size,in_units=hidden_size,flatten=False)
            #self.state = self.rnn.begin_state(batch_size=22)
    def forward(self, inputs,state, *args, **kwargs):
        #print("input",inputs.shape)
        # input shape 1*3*36*248 (batch_size, in_channels, height, width)`
        X = self.cnn(inputs)
        X = self.polling(X)
        #X = self.act(X)
        #X = X.squeeze(axis=1)
        # print("cnn",X.shape)
        #X = X.resize((1,1,9,62))
        X = X.transpose((3,0,1,2))
        # print("transpose",X.shape)
        #print(X.shape,state.shape)
        output = []
        for time_step in X:
            #print("time_step")
            X,state = self.rnn(time_step,state)
            output.append(X)

        X = mx.nd.concat(*output,dim=0)
        # print("concat shape ",X.shape)
        X = self.fc(X)
        #X = self.fc(X.reshape((-1, self.num_hidden)))
        # print("fc", X.shape)

        return X,state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
# -1 is background
char_dict = {'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','-1':'10','d':'11','+':'12'}
id2char = [0,1,2,3,4,5,6,7,8,9,'_']
vocab_size = len(char_dict)
char2id = []

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
    print('str_list',str_list)
    index = 0
    length = (len(str_list))
    label_list = []
    # add -1 at repeat num ,such as : 00-->0-*-0
    while index < length:
        #print (index)
        if index + 1 < length and str_list[index] == str_list[index+1] :
            label_list.append(int(char_dict[str_list[index]]))
            label_list.append(-1)
        else:
            label_list.append(int(char_dict[str_list[index]]))
        index = index + 1
    #print(label_list)
    if len(label_list) < seq_length:
        le = seq_length - len(label_list)
        other = [-1]*le
        # print("other ",other)
        label_list.extend(other)
    print (label_list)
    return label_list
def data_iter(img_path):
    """
    :return: data ,label
    """
    for root,dir,files in os.walk(img_path):
        for file in files:
            # print("img ",file)
            assert os.path.exists(root+file)," file not exists {}".format(root+file)
            data = mx.image.imread(root+file)
            data = data.astype('float32') / 255.0
            data = data - 0.5
            # print("data shape ", data.shape)
            data = data.reshape(1, 128, 60, 3)
            # print("data shape ", data.shape)
            data = nd.transpose(data, (0, 3, 1, 2))
            #get the label from the file
            label_str = file.split("=")[-1].replace(".jpg","")
            label_str = label_str.replace(".","")
            label_str = label_str.replace("+", "")
            # print("after data shape :", data.shape)
            label_list = to_ctc_format(label_str, 56)
            # print("label_list",label_list)
            #d = nd.one_hot(label_list)
            #print("d  ",d)
            label = nd.array([label_list])

            d = nd.one_hot(label,11)
            d = nd.reshape(d,shape=(11,56))
            # print("d  ",d)
            # print("d  ", d.shape)
            yield data,label

def train(data_iter):
    lstm = OCRLSTM()
    lstm.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())

    loss = gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    trainer = gluon.Trainer(lstm.collect_params(), 'sgd', {'learning_rate': 0.001})
    state = lstm.begin_state(batch_size=1)
    global_step = 0

    for epoch in range(100):
        print("epoch ",epoch)
        for sample in data_iter:
            data = sample[0]
            label = sample[1]
            train_loss = .0
            with autograd.record():
                # print("data ",state)
                output, state = lstm(data, state)
                # output = nd.expand_dims(output, axis=1)
                output = output.transpose((1, 0, 2))
                # label = nd.expand_dims(label, axis=1)
                # label = label.reshape((1,4))
                # print("output ", output.shape, label.shape)
                L = loss(output, label)
            L.backward()
            train_loss = nd.mean(L).asscalar()
            # sw.add_scalar(tag="loss",value=train_loss,global_step=global_step)
            global_step = global_step + 1
            # if epoch == 1 :
            #     sw.add_graph(net)
            trainer.step(1, ignore_stale_grad=True)

            if (epoch % 100 == 0):
                print('train_loss %.4f' % (train_loss))
                # print('output max', output.argmax(axis=2))
            predict(data,state)
import numpy

def predict(data,state):
    lstm = OCRLSTM()
    lstm.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())

    op, state1 = lstm(data, state)
    #print("op ",op.shape)
    op = nd.reshape(op, (56, 11))
    #print("op ", op)
    tt = mx.nd.softmax(op)
    tt2 = tt.asnumpy().argmax(axis=1)
    #print("tt2",tt)
    tag = [d for d in tt2]
    # rec = ctc_label(tt.asnumpy().argmax( axis=1))
    # print("tag : ",tag)
    str_num = ''
    print(ctc_label(tag))
    for i in tag:
        # if i in char_dict.keys():
        str_num = str_num + " " + str(i)
    print("result ", str_num)
if __name__ == "__main__":
    from mxboard import SummaryWriter

    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    from mxnet import autograd, gluon, image, init, nd
    import mxnet as mx

    img_path = "/home/abner/dnn/project/tf/Attention-OCR/data/cut1/"

    data = data_iter(img_path)
    data_ite = []
    for d,l in data:
        temp = []
        temp.append(d)
        temp.append(l)
        data_ite.append(temp)

    train(data_ite)

        #prediction = [p - 1 for p in rec]
        # print("prediction : ",prediction)

    #export the model

    #net.export("mod1")
    # net = gluon.nn.SymbolBlock.imports('mod1-symbol.json', ['data'], param_file='mod1-0000.params',   ctx=mx.cpu())
    # output = net(data)
    # output = output[0].reshape((-1, output.shape[-1]))
    # print (output.shape )
    # print(output.argmax(axis=1))
    # print(len(output.argmax(axis=1)))


