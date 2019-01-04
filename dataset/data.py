from mxnet import nd
#data set process
import mxnet as mx
ctx = mx.cpu(0)
with open("../data/train.txt") as f:
    time_machine = f.read()
#time_machine = time_machine[:-38000]
print (time_machine)
character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    #print(char)
    character_dict[char] = e
    #print(character_dict[char] )
time_numerical = [character_dict[char] for char in time_machine]

def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result
def get_data():
    batch_size = 32
    seq_length = 64
    # -1 here so we have enough characters for labels later
    num_samples = (len(time_numerical) - 1) // seq_length
    print(time_numerical[:seq_length * num_samples])
    dataset = one_hots(time_numerical[:seq_length * num_samples]).reshape((num_samples, seq_length, vocab_size))
    num_batches = len(dataset) // batch_size
    train_data = dataset[:num_batches * batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
    #print(train_data)
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 1, 2)

    labels = one_hots(time_numerical[1:seq_length * num_samples + 1])
    #print(len(labels))
    train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
    train_label = nd.swapaxes(train_label, 1, 2)
    #print(len(train_data),len(train_label))

    return train_data,train_label

if __name__=='__main__':
    d,l = get_data()
    print(d[0][1][2])
    print(l[0][1][2])

    #print("l,length",l,len(l))