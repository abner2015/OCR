#coding:utf-8
from mxnet import nd

def to_ctc_format(label,seq_length):
    #seq_length = 176
    str = label
    str_list = [x for x in str]
    print('str_list',str_list)
    index = 0
    length = (len(str_list))
    label_list = []
    # add -1 at repeat num ,such as : 00-->0-*-0
    while index < length:
        print (index)
        if index + 1 < length and str_list[index] == str_list[index+1] :
            label_list.append(str_list[index])
            label_list.append(-1)
        else:
            label_list.append(str_list[index])
        index = index + 1
    #print(label_list)
    if len(label_list) < seq_length:
        le = seq_length - len(label_list)
        other = [-1]*le
        print("other ",other)
        label_list.extend(other)
    #print (label_list)
    return label_list

if __name__ =='__main__':

   to_ctc_format("0001011",20)