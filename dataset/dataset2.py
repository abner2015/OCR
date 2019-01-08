import random
from mxnet import nd

def data_iter_random(corpus_indeces,batch_size,num_steps,ctx=None):
    num_examples = (len(corpus_indeces) - 1) // num_steps
    epoch_size = num_examples // batch_size
    examplex_indices = list(range(num_examples))

    random.shuffle(examplex_indices)

    def _data(pos):
        return corpus_indeces[pos:pos + num_steps]


    for i in range(epoch_size):
        print("i ",i)
        i = i * batch_size
        batch_indices = examplex_indices[i:i+batch_size]

        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps+1) for j in batch_indices]
        yield nd.array(X,ctx),nd.array(Y,ctx)

def to_onehot(X, size):  # 本函数已保存在 d2lzh 包中方便以后使用。
    return [nd.one_hot(x, size) for x in X.T]
if __name__ == "__main__":
    my_seq = list(range(60))
    for X,Y in data_iter_random(my_seq,batch_size=2,num_steps=60):
        print('X: ',type(X[0][0]), '\nY',Y)