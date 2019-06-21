import cv2
#from skimage.io import imread, imshow
#import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
import tensorflow as tf

capture = cv2.VideoCapture(0)
capture.set(3, 320)#width
capture.set(4, 240)#height

model = load_model('unet_model.h5')

while True:
    okay, im = capture.read()
    im = cv2.resize(image,(128, 128))
    
    input_img = np.expand_dims(im, axis=0)

    test_predictions = model.predict(input_img)
    #preds_train_t = (test_predictions > 0.5).astype(np.uint8)

    test_im = np.squeeze(test_predictions)#remove the dimentions in each ends

    cv2.imshow('LANE Tracker', test_im)
    key = cv2.waitKey(33)
    if key == 27:
        break

cv2.destroyAllWindows()    


