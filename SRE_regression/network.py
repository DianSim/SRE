import tensorflow as tf
from tensorflow.keras import layers, models
from config import config

# config_model = config['model']
config_model = config['model_name']
config_model_params = config['model_params']
frame_len = int(config['frame_length']*config["sample_rate"]/1000) # 560

class My_Model(tf.keras.models.Model):
    def __init__(self):
        super(My_Model, self).__init__() 

        # gave seq_len None to handle varibale length sequences
        input = layers.Input(shape=(None, 560))
        x = layers.LSTM(units=64, return_sequences=False)(input)
        # x = layers.BatchNormalization()(x)
        x = layers.Dense(1)(x)
        clipped_output = tf.clip_by_value(x, clip_value_min=0, clip_value_max=23) # change to Relu
        self.model = models.Model(inputs=input, outputs=clipped_output)

    def call(self, x):
        return self.model(x)