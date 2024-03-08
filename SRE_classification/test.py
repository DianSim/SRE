import os
import tensorflow as tf
from config import config
import numpy as np
from scipy.stats import pearsonr
import tensorflow_addons as tfa
from sklearn.metrics import f1_score


class Test:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tfa.metrics.PearsonsCorrelation()] 
            )

        model_name = config['model_name']
        models_dir = './models'
        checkpoint_dir = os.path.join(models_dir, model_name, 'checkpoints')

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(latest_checkpoint)
            print(f"Loaded latest checkpoint: {latest_checkpoint}")
            print('test: ', self.model.evaluate(self.dataset))
        else:
            print(f"There's no saved checkpoint found in the {checkpoint_dir} dir.")

        
        