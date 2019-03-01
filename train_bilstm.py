

import os
import random
import string
import time

import matplotlib.pyplot as plt
from mxboard import SummaryWriter
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.model_zoo.vision import resnet34_v1
import numpy as np
from skimage import transform as skimage_tf
from skimage import exposure,io

from tqdm import tqdm
from src.cnn_bilstm import CNNBiLSTM
from mxnet.gluon import data
import cv2
from skimage.transform import rescale, resize
from skimage import color
from src.dataset.ocr_dataset import OCRDataset


np.seterr(all='raise')
alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

random_y_translation, random_x_translation = 0.03, 0.03
random_y_scaling, random_x_scaling = 0.1, 0.1
random_shearing = 0.7
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

epochs = 130
learning_rate = 0.001
batch_size = 3

max_seq_len = 32
print_every_n = 5
send_image_every_n = 5

num_downsamples = 2
resnet_layer_id = 4
lstm_hidden_states = 512
lstm_layers = 2

ctc_loss = gluon.loss.CTCLoss(weight=0.2)
best_test_loss = 10e5

def transform(image, label):
    '''
    This function resizes the input image and converts so that it could be fed into the network.
    Furthermore, the label (text) is one-hot encoded.
    '''

    #image = np.expand_dims(image, axis=0).astype(np.float32)
    #print(image[0,0,0])

    #image = np.expand_dims(image, axis=0).astype(np.float32)
    image = image.astype(np.float32)
    if (image[0, 0, 0] > 1).all():
        image = image /255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    label_encoded = np.zeros(max_seq_len, dtype=np.float32 ) -1
    i = 0
    for word in label:
        word = word.replace("&quot", r'"')
        word = word.replace("&amp", r'&')
        word = word.replace('";', '\"')
        for letter in word:
            label_encoded[i] = alphabet_dict[letter]
            i += 1

    #print(type(image))

    image = np.reshape(image, (3, 128, 64))
    image = np.resize(image,(1, 128, 64))

    return image, label_encoded

def augment_transform(image, label):
    '''
    This function randomly:
        - translates the input image by +-width_range and +-height_range (percentage).
        - scales the image by y_scaling and x_scaling (percentage)
        - shears the image by shearing_factor (radians)
    '''
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    sx = random.uniform(1. - random_y_scaling, 1. + random_y_scaling)
    sy = random.uniform(1. - random_x_scaling, 1. + random_x_scaling)

    s = random.uniform(-random_shearing, random_shearing)

    gamma = random.uniform(0.001, 2)
    image = exposure.adjust_gamma(image, gamma)
    st = skimage_tf.AffineTransform(scale=(sx, sy),
                                    shear=s,
                                    translation=(tx *image.shape[1], ty *image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)

    return transform(augmented_image *255., label)


import cv2
import numpy as np


def draw_text_on_image(images, text):
    output_image_shape = (images.shape[0], images.shape[1], images.shape[2] * 2,
                          images.shape[3])  # Double the output_image_shape to print the text in the bottom

    output_images = np.zeros(shape=output_image_shape)
    for i in range(images.shape[0]):
        white_image_shape = (images.shape[2], images.shape[3])
        white_image = np.ones(shape=white_image_shape) * 1.0
        text_image = cv2.putText(white_image, text[i], org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                 color=0.0, thickness=1)
        output_images[i, :, :images.shape[2], :] = images[i]
        output_images[i, :, images.shape[2]:, :] = text_image
    return output_images

def decode(prediction):
    '''
    Returns the string given one-hot encoded vectors.
    '''
    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[ i +1] and word[-1] != -1:  # Hack to decode label as well
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(result)
    words = [''.join(word) for word in results]
    return words

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, is_train):
    total_loss = nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        with autograd.record(train_mode=is_train):
            output = network(x)
            loss_ctc = ctc_loss(output, y)

        if is_train:
            loss_ctc.backward()
            trainer.step(x.shape[0])

        if i == 0 and e % send_image_every_n == 0 and e > 0:
            predictions = output.softmax().topk(axis=2).asnumpy()
            decoded_text = decode(predictions)
            output_image = draw_text_on_image(x.asnumpy(), decoded_text)
            output_image[output_image < 0] = 0
            output_image[output_image > 1] = 1
            print("{} first decoded text = {}".format(print_name, decoded_text[0]))
            with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
                sw.add_image('bb_{}_image'.format(print_name),output_image,global_step=e)


        total_loss += loss_ctc.mean()

    epoch_loss = float(total_loss.asscalar())/len(dataloader)

    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    return epoch_loss

if __name__ == '__main__':
    log_dir = "E:/project/OCR/logs/handwriting_recognition"
    checkpoint_dir = "E:/project/OCR/model_checkpoint"
    checkpoint_name = "handwriting.params"
    train_path = "E:/project/OCR/data/train/"
    test_path = "E:/project/OCR/data/train/"
    ###
    train_ds = OCRDataset(train_path)
    print("Number of training samples: {}".format(len(train_ds)))

    test_ds = OCRDataset(train_path)
    print("Number of testing samples: {}".format(len(test_ds)))
    train_data = gluon.data.DataLoader(train_ds.transform(transform), batch_size, shuffle=True,
                                       last_batch="rollover", num_workers=0)

    test_data = gluon.data.DataLoader(test_ds.transform(transform), batch_size, shuffle=True, last_batch="keep",num_workers=0)  # , num_workers=multiprocessing.cpu_count()-2)

    ###
    net = CNNBiLSTM(num_downsamples=num_downsamples, resnet_layer_id=resnet_layer_id,
                    rnn_hidden_states=lstm_hidden_states, rnn_layers=lstm_layers, max_seq_len=max_seq_len, ctx=ctx)

    net.hybridize()
    #net.initialize(init=mx.init.Xavier(), force_reinit=True)
    hybridlayer_params = {k: v for k, v in net.collect_params().items()}
    print("net ",net)
    for key, value in hybridlayer_params.items():
        print('{} = {}\n'.format(key, value.shape))
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

    for e in range(epochs):
        train_loss = run_epoch(e, net, train_data, trainer, log_dir, print_name="train", is_train=True)
        test_loss = run_epoch(e, net, test_data, trainer, log_dir, print_name="test", is_train=False)
        if test_loss < best_test_loss:
            print("Saving network, previous best test loss {:.6f}, current test loss {:.6f}".format(best_test_loss,
                                                                                                    test_loss))
            net.save_parameters(os.path.join(checkpoint_dir, checkpoint_name))
            best_test_loss = test_loss

        if e % print_every_n == 0 and e > 0:
            print("Epoch {0}, train_loss {1:.6f}, test_loss {2:.6f}".format(e, train_loss, test_loss))