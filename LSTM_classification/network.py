import tensorflow as tf
from tensorflow.keras import layers, models
from config import config
from augmentation import WhiteNoiseAugmentation

# config_model = config['model']
config_model = config['model_name']
config_model_params = config['model_params']
frame_len = int(config['frame_length']*config["sample_rate"]/1000) # 560

class My_Model(tf.keras.models.Model):
    def __init__(self):
        super(My_Model, self).__init__() 

        # gave seq_len None to handle varibale length sequences
        input = layers.Input(shape=(None, 560))
        x = layers.LSTM(units=128, return_sequences=False)(input)
        output = layers.Dense(25, activation='softmax')(x)
        self.model = models.Model(inputs=input, outputs=output)

    def call(self, x):
        return self.model(x)