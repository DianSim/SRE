import tensorflow as tf
import keras


tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

@keras.saving.register_keras_serializable()
class WhiteNoiseAugmentation(tf.keras.layers.Layer):
    def __init__(self, factor=0.5):
        super().__init__()
        self.factor = factor

    def call(self, inputs, training=None):
        if not training:
            return inputs
       
        augmented_batch = []
        for x in inputs:
            if  tf.random.uniform([]) < self.factor:
                noise_std = tf.random.uniform(shape=[], minval=0.4, maxval=1)
                noise = tf.random.normal(shape=tf.shape(inputs)[1:], mean=0.0, stddev=noise_std)
                noisy_signal = x + noise
                augmented_batch.append(noisy_signal)
            else:
                augmented_batch.append(x)
        return tf.stack(augmented_batch)