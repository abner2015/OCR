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
        #print("i ",i)
        i = i * batch_size
        batch_indices = examplex_indices[i:i+batch_size]

        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps+1) for j in batch_indices]

        #print(X)
        yield nd.array(X,ctx),nd.array(Y,ctx)

def to_onehot(X, size):  # 本函数已保存在 d2lzh 包中方便以后使用。
    return [nd.one_hot(x, size) for x in X.T]
if __name__ == "__main__":
    my_seq = list(range(60))

    for X,Y in data_iter_random(my_seq,batch_size=5,num_steps=2):
        #X = nd.one_hot(X[0], 60)
        print("before reshape",X)
        for x in X.T:
            print("x",x)
            print(nd.one_hot(x,60))
        # print("trans reshape",X.T)
        #
        # print(to_onehot(X.T,60))
        # #X = X.reshape((1,5,2))
        print(Y.T.reshape((-1)))
        a = [nd.one_hot(x, 60) for x in Y.T]
        print(Y)
        print("concat Y",nd.concat(*Y,dim=0))
        #print(a)
        #print('X: ',X[0], '\nY',Y)