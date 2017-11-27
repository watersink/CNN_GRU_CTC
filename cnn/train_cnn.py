#coding:utf-8
from captcha.image import ImageCaptcha
import numpy as np
import random
import string


from keras.models import *
from keras.layers import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

characters = string.digits + string.ascii_uppercase+ string.ascii_lowercase
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

width, height, n_len, n_class = 170, 80, 4, len(characters)


def generateImg():
    generator=ImageCaptcha(width=width, height=height)
    random_str = ''.join([random.choice(characters) for j in range(5)])
    img_name= "{}.png".format(random_str)
    generator.write(random_str,img_name)

def gen(batch_size=32):
	#generate data on the fly
	#X numpy 
	#y list，num of list is n_len,in list is numpy,size is batch_size*n_class
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

		
def init_model():
	input_tensor = Input((height, width, 3))
	x = input_tensor
	for i in range(4):
		x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
		x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
		x = MaxPooling2D((2, 2))(x)

	x = Flatten()(x)
	x = Dropout(0.25)(x)
	x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
	model = Model(input=input_tensor, output=x)
	return model

def train_model(model):	
	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])		  
	model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
                    validation_data=gen(), nb_val_samples=1280)				
	model.save('cnn.h5')
	
	
if __name__=="__main__":
	generateImg()
    #model=init_model()
    #train_model(model)