#encoding:utf-8
import cv2
import os
import random
import string
import numpy as np
from captcha.image import ImageCaptcha

from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.callbacks import *

characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
width, height, n_len, n_class = 170, 80, 4, len(characters)+1


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def init_model():
    rnn_size = 128

    input_tensor = Input((width, height, 3))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
    gru1_merged = merge([gru_1, gru_1b], mode='sum')

    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
    x = merge([gru_2, gru_2b], mode='concat')
    x = Dropout(0.25)(x)
    x = Dense(n_class, init='he_normal', activation='softmax')(x)

    base_model = Model(input=input_tensor, output=x)

    return conv_shape,base_model



def gen(batch_size=28):
    global conv_shape
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*n_len], np.ones(batch_size)


def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = gen(128)
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        #out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :4]
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :]

        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    accuracy= batch_acc / batch_num
    print("accuracy:",accuracy)
	


def test(model):
    [X_test, y_test, _, _], _  = next(gen(1))
    y_pred = model.predict(X_test)
    y_pred = y_pred[:,2:,:]
    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]
    out = ''.join([characters[x] for x in out[0]])
    y_true = ''.join([characters[x] for x in y_test[0]])
    print(('pred:' + str(out) + '\ntrue: ' + str(y_true)))
	
def test_from_img(model):
    base_dir="../images/"
    for img_name in os.listdir(base_dir):
        img = cv2.imread(base_dir+img_name)
        img = img.transpose(1, 0, 2)
        img = img.reshape(1, width, height, 3)
        y_pred = model.predict(img)
        y_pred = y_pred[:, 2:, :]
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
        # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
        out = ''.join([characters[x] for x in out[0]])
        print('real: %s pred:%s' % (img_name.split(".")[0], str(out)))






if __name__=="__main__":
    conv_shape,model=init_model()
    model.load_weights("ctc.h5",by_name=True)

    print("evaluate:......")
    evaluate(model)

    print("test:......")
    test(model)
	
    print("test from img:......")
    test_from_img(model)