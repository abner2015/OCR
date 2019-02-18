from mxnet import gluon
import mxnet as mx
from mxnet.gluon.model_zoo.vision import resnet34_v1
from src.encoderlayer import EncoderLayer

alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}
class CNNBiLSTM(gluon.HybridBlock):
    '''
    recognise the image
    num_downsamples:int,default 2
        the number of times to downsample the image features.Each time the features are downsampled, a new LSTM is created.
    resnet_layer_id:int,default 4
        The layer ID to obtain features from the resnet34
    lstm_hidden_states:int default 200
        The number of hidden states used in the LSTMs
    lstm_layers: int, default 1

    '''
    FEATURE_EXTRACTOR_FILTER = 64
    def __init__(self,num_downsamples=2,resnet_layer_id = 4,rnn_hidden_states=200,rnn_layer=1,max_seq_len=100,ctx=mx.gpu(0),**kwargs):
        super(CNNBiLSTM,self).__init__(**kwargs)
        self.p_dropout = 0.5
        self.num_downsamples = num_downsamples
        self.max_seq_len = max_seq_len
        self.ctx = ctx
        with self.name_scope():
            self.body = self.get_body(resnet_layer_id=resnet_layer_id)
            self.encoders = gluon.nn.HybridSequential()
            with self.encoders.name_scope():
                for i in range(self.num_downsamples):
                    encoder = self.getencoder(rnn_hidden_state=rnn_hidden_states,rnn_layers=rnn_layers,max_seq_len=max_seq_len)
                    self.encoders.add(encoder)
                self.decoder = self.get_decoder()
                self.downsampler = self.get_down_sampler(self.FEATURE_EXTRACTOR_FILTER)
    def get_down_sampler(self,num_filters):
        '''
        downsample  feaures  1/2 size
        :param num_filters: int
            To select the number of filters in used the downsampling convolutional layer.
        :return:
        network: gulon.nn.HybridSequential


        '''
        out = gluon.nn.HybridSequential()
        with out.name_scope():
            for _ in range(2):
                out.add(gluon.nn.Conv2D(num_filters,3,strides=1,padding=1))
                out.add(gluon.nn.BatchNorm(in_channels=num_filters))
                out.add(gluon.nn.Activation('relu'))
            out.add(gluon.nn.MaxPool2D(2))
            out.collect_params().initalize(mx.init.Normal(),ctx = self.ctx)
        out.hybridize()
        return out

    def get_body(self,resnet_layer_id):
        pretrained = resnet34_v1(pretrained=True,ctx=self.ctx)
        pretrained_2 = resnet34_v1(pretrained=True,ctx=mx.cpu(0))
        first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
        body = gluon.nn.HybridSequential()
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64,kernel_size=(7,7),padding=(3,3),strides=(2,2),in_channels=1,use_bias=False)
            first_layer.initialize(mx.init.Xavier(),ctx=self.ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:-resnet_layer_id])
        return body

    def get_encoder(self,rnn_hidden_states,rnn_layers,max_seq_len):
        encoder = gluon.nn.HybridSequential()
        with encoder.name_scope():
            encoder.add(EncoderLayer(hidden_states=rnn_hidden_states,rnn_layers=rnn_layers,max_seq_len=max_seq_len))
            encoder.add(gluon.nn.Dropout(self.p_dropout))
        encoder.collect_params().initialize(mx.init.Xavier(),ctx=self.ctx)
        return encoder

    def get_decoder(self):
        alphabet_size = len(alphabet_encoding) + 1
        decoder = mx.gluon.nn.Dense(units=alphabet_size,flatten=False)
        decoder.collect_params().initialize(mx.init.Xavier(),ctx=self.ctx)
        return decoder

    def hybrid_forward(self, F, x):
        features = self.body(x)
        hidden_states = []
        hs = self.encoders[0](features)
        hidden_states.append(hs)
        for i,_ in enumerate(range(self.num_downsamples - 1)):
            features = self.downsampler(features)
            hs = self.encoders[i+1](features)
            hidden_states.append(hs)
        hs = F.concat(*hidden_states,dim=2)
        output = self.decoder(hs)
        return output


