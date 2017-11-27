#coding:utf-8
import cv2
import string
import random
import numpy as np
from tqdm import tqdm
from captcha.image import ImageCaptcha

from keras.models import load_model


characters = string.digits + string.ascii_uppercase+ string.ascii_lowercase
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
width, height, n_len, n_class = 170, 80, 4, len(characters)

def gen(batch_size=32):
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

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])		
		
def evaluate(model, batch_num=100):
    batch_acc = 0
    generator = gen()
    #for i in tqdm(range(batch_num)):
    for i in range(batch_num):
        X, y = next(generator)
        y_pred = model.predict(X)
        batch_acc += np.mean(np.asarray(list(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))))
    print(float(batch_acc) / batch_num)

	
def test(model):
	X, y = next(gen(1))
	y_pred = model.predict(X)
	print('real: %s\npred:%s'%(decode(y), decode(y_pred)))

def test_from_img(model):
	img_name="../images/dspa.png"
	img=cv2.imread(img_name)
	img = img.reshape(1, height, width, 3)
	y_pred = model.predict(img)
	print('real: %s pred:%s'%(img_name.split("/")[2].split(".")[0], decode(y_pred)))


if __name__=="__main__":
	model=load_model("cnn.h5")
	
	print("evaluate:......")
	evaluate(model)
	
	print("test:......")
	test(model)
	
	print("test from img:......")
	test_from_img(model)
	
