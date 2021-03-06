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

import mxnet as mx

class TestSet(mx.gluon.data.Dataset):
    def __init__(self):
        self.x = mx.nd.zeros((10, 5))
        self.y = mx.nd.arange(10)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return 10
import os
import numpy as np
from skimage import exposure,io
alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

class OCRDataset(mx.gluon.data.Dataset):
    def __init__(self,root):
        self.root = root
        self.data = []
        self.label = []

        for root1,dir,files in os.walk(self.root):
            for file in files:
                image_path = root1+file
                #image = cv2.imread(image_path)
                image = io.imread(image_path)
                #image = mx.image.imread(image_path)
                label = file.split("=")[-1].replace(".jpg", "")
                label_encoded = np.zeros(160, dtype=np.float32) - 1
                i = 0
                for word in label:
                    word = word.replace("&quot", r'"')
                    word = word.replace("&amp", r'&')
                    word = word.replace('";', '\"')
                    for letter in word:
                        label_encoded[i] = alphabet_dict[letter]
                        i += 1
                self.data.append(image)

                self.label.append(label)

    def __getitem__(self, idx):
        print("idx ",idx)
        print("label ", self.label[idx])
        print("__getitem__",np.array(self.data[idx]).shape,np.array(self.label[idx]).shape)
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

if __name__ =='__main__':
    #OCRDataset("E:/project/OCR/data/train/")
    for i, data in enumerate(mx.gluon.data.DataLoader(TestSet(), batch_size=2)):
        print(data)
   #to_ctc_format("0001011",20)