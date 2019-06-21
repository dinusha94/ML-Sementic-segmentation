import cv2
#from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
import tensorflow as tf

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

input_img =  cv2.imread("data_images/1413.jpg", cv2.IMREAD_COLOR)
input_img = np.expand_dims(input_img, axis=0)

model = load_model('unet_model.h5')


test_predictions = model.predict(input_img)
#preds_train_t = (test_predictions > 0.5).astype(np.uint8)


test_im = np.squeeze(test_predictions)#remove the dimentions in each ends


plt.imshow(test_im, cmap=plt.cm.gray)
plt.show()


