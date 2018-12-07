from mxnet import nd
import mxnet as mx
from src.lstm import LSTM
def main(lstm):
    epochs = 2000
    moving_loss = 0.
    learning_rate = 2.0

    for e in range(epochs):
        if ((e+1) %100 == 0):
            learning_rate = learning_rate /2.0
        h = nd.zeros(shape=(batch_size,num_hidden),ctx)
        c = nd.zeros(shape=(batch_size,num_hidden),ctx)

        for i in range(num_batches):
            data_one_hot = train_data[i]
            label_one_hot = train_label[i]
            with autograd.record():
                outputs,h,c = lstm.lstm_rcnn(data_one_hot,h,c)
                loss = lstm.average_ce_loss(outputs,label_one_hot)
                loss.backward()
            lstm.SGD(learning_rate=learning_rate)

            if ( i == 0 ) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99*moving_loss + .01*nd.mean(loss).asscalar()
        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("The Time Ma", 1024, temperature=.1))
        print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))
if __name__ == '__main__':
    lstm = LSTM
    main(lstm)