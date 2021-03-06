from mxnet import gluon

class EncoderLayer(gluon.HybridBlock):
    '''
    1.Get image features from CNN
    2.Transposed the features so that the LSTM sllices (sequentially)
    '''
    def __init__(self,hidden_states=200,rnn_layers=1,max_seq_len=100,input_size=3584,**kwargs):
        self.max_seq_len = max_seq_len
        super(EncoderLayer,self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = gluon.rnn.LSTM(hidden_states,rnn_layers,bidirectional=True,input_size=input_size)
    def hybrid_forward(self, F, x):
        #print("infer", x.infer_shape(x))
        x = x.transpose((0,3,1,2))
        x = x.flatten()
        x = x.split(num_outputs=self.max_seq_len,axis=1)#(SEQ_LEN,N,CHANNELS)
        #mx.viz.plot_network(x).view()
        x = F.concat(*[elem.expand_dims(axis=0) for elem in x],dim=0)
        x = self.lstm(x)
        #mx.viz.plot_network(x).view()
        x = x.transpose((1,0,2)) #(N,SEQ_LEN,HIDDEN_UNITS)

        return x



