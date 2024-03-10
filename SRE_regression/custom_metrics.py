from keras import backend as K
from tensorflow import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)